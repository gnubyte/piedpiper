//! Minimal Vulkan compute runtime: one device and queue, one storage-buffer memory pool bound
//! to every kernel (bindings 0 and 1 alias it as float[] and uint[]), 32 words of push
//! constants per dispatch, and one command buffer that is recorded, submitted and waited on.

use ash::vk;
use std::ffi::CStr;

pub const PUSH_WORDS: usize = 32;

pub type Res<T> = Result<T, String>;

fn vkerr(what: &str) -> impl Fn(vk::Result) -> String + '_ {
    move |e| format!("{what}: {e:?}")
}

pub struct Buffer {
    pub buf: vk::Buffer,
    mem: vk::DeviceMemory,
    pub bytes: u64,
    ptr: *mut u8,
}

impl Buffer {
    fn host(&self) -> *mut u8 {
        assert!(!self.ptr.is_null(), "buffer is not host-visible");
        self.ptr
    }

    pub fn write_u32(&self, word: usize, data: &[u32]) {
        assert!(((word + data.len()) * 4) as u64 <= self.bytes);
        // SAFETY: in bounds of the persistent host mapping; the GPU is idle between submits.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), (self.host() as *mut u32).add(word), data.len()) }
    }

    pub fn write_f32(&self, word: usize, data: &[f32]) {
        assert!(((word + data.len()) * 4) as u64 <= self.bytes);
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), (self.host() as *mut f32).add(word), data.len()) }
    }

    pub fn f32s(&self, word: usize, len: usize) -> &[f32] {
        assert!(((word + len) * 4) as u64 <= self.bytes);
        unsafe { std::slice::from_raw_parts((self.host() as *const f32).add(word), len) }
    }
}

pub struct Kernel {
    pipe: vk::Pipeline,
    pub name: &'static str,
}

const PROFILE_SLOTS: u32 = 8192;

/// What follows each dispatch. `Compute` is enough for kernel-to-kernel dependencies (copies
/// always get their own transfer barriers); `Full` and `Off` exist to measure the cost.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Barriers {
    Full,
    Compute,
    Off,
}

pub struct Gpu {
    _entry: ash::Entry,
    instance: ash::Instance,
    pub device: ash::Device,
    queue: vk::Queue,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    pub name: String,
    pub driver: String,
    pub limits: vk::PhysicalDeviceLimits,
    pool: vk::CommandPool,
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    dsl: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    dpool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    qpool: vk::QueryPool,
    ts_period_ns: f64,
    pipes: Vec<vk::Pipeline>,
    buffers: Vec<(vk::Buffer, vk::DeviceMemory)>,
    /// Dispatches recorded since creation (for overhead accounting).
    pub dispatches: u64,
    /// Submissions since creation.
    pub submits: u64,
    /// Total GPU time (timestamps) and wall time spent in submit + wait.
    pub gpu_seconds: f64,
    pub wait_seconds: f64,
    pub barriers: Barriers,
    /// Poll the fence instead of sleeping in vkWaitForFences (lower latency, burns a core).
    pub spin: bool,
    /// Per-dispatch timestamps, accumulated per kernel name (GPU seconds, count).
    pub profile: Option<std::collections::BTreeMap<&'static str, (f64, u64)>>,
    ppool: vk::QueryPool,
    plabels: Vec<&'static str>,
}

impl Gpu {
    /// Opens the first discrete GPU (or `PP_GPU=<index>`), checking that it supports the float
    /// controls the bit-exact kernels need.
    pub fn new() -> Res<Gpu> {
        unsafe {
            let entry = ash::Entry::load().map_err(|e| format!("cannot load the Vulkan loader: {e}"))?;
            let app = vk::ApplicationInfo::default()
                .application_name(c"piedpiper-gpu-experiment")
                .api_version(vk::API_VERSION_1_3);
            let instance = entry
                .create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)
                .map_err(vkerr("vkCreateInstance"))?;
            let pdevs = instance.enumerate_physical_devices().map_err(vkerr("enumerate devices"))?;
            if pdevs.is_empty() {
                return Err("no Vulkan device".into());
            }
            let pick = match std::env::var("PP_GPU").ok().and_then(|s| s.parse::<usize>().ok()) {
                Some(i) => *pdevs.get(i).ok_or("PP_GPU out of range")?,
                None => *pdevs
                    .iter()
                    .find(|&&p| instance.get_physical_device_properties(p).device_type == vk::PhysicalDeviceType::DISCRETE_GPU)
                    .unwrap_or(&pdevs[0]),
            };
            let mut v12 = vk::PhysicalDeviceVulkan12Properties::default();
            let props = {
                let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut v12);
                instance.get_physical_device_properties2(pick, &mut p2);
                p2.properties
            };
            let name = props.device_name_as_c_str().map(|c| c.to_string_lossy().into_owned()).unwrap_or_default();
            let driver = format!(
                "{} {}",
                v12.driver_name_as_c_str().map(|c| c.to_string_lossy().into_owned()).unwrap_or_default(),
                v12.driver_info_as_c_str().map(|c| c.to_string_lossy().into_owned()).unwrap_or_default()
            );
            let need = [
                ("shaderDenormPreserveFloat32", v12.shader_denorm_preserve_float32),
                ("shaderRoundingModeRTEFloat32", v12.shader_rounding_mode_rte_float32),
                ("shaderSignedZeroInfNanPreserveFloat32", v12.shader_signed_zero_inf_nan_preserve_float32),
            ];
            for (n, v) in need {
                if v != vk::TRUE {
                    return Err(format!("{name}: {n} not supported, so results could differ from the CPU"));
                }
            }
            if props.limits.max_push_constants_size < (PUSH_WORDS * 4) as u32 {
                return Err("maxPushConstantsSize < 128".into());
            }
            let qfams = instance.get_physical_device_queue_family_properties(pick);
            let qf = qfams
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE) && q.timestamp_valid_bits > 0)
                .ok_or("no compute queue with timestamps")? as u32;
            let prio = [1.0f32];
            let qci = [vk::DeviceQueueCreateInfo::default().queue_family_index(qf).queue_priorities(&prio)];
            let device = instance
                .create_device(pick, &vk::DeviceCreateInfo::default().queue_create_infos(&qci), None)
                .map_err(vkerr("vkCreateDevice"))?;
            let queue = device.get_device_queue(qf, 0);
            let mem_props = instance.get_physical_device_memory_properties(pick);
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                        .queue_family_index(qf),
                    None,
                )
                .map_err(vkerr("command pool"))?;
            let cb = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(vkerr("command buffer"))?[0];
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(vkerr("fence"))?;
            let bindings = [0u32, 1].map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            });
            let dsl = device
                .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)
                .map_err(vkerr("set layout"))?;
            let ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size((PUSH_WORDS * 4) as u32)];
            let dsls = [dsl];
            let layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&dsls).push_constant_ranges(&ranges),
                    None,
                )
                .map_err(vkerr("pipeline layout"))?;
            let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 2 }];
            let dpool = device
                .create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None)
                .map_err(vkerr("descriptor pool"))?;
            let set = device
                .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(dpool).set_layouts(&dsls))
                .map_err(vkerr("descriptor set"))?[0];
            let qpool = device
                .create_query_pool(
                    &vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2),
                    None,
                )
                .map_err(vkerr("query pool"))?;
            let ppool = device
                .create_query_pool(
                    &vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(PROFILE_SLOTS),
                    None,
                )
                .map_err(vkerr("query pool"))?;
            Ok(Gpu {
                _entry: entry,
                instance,
                device,
                queue,
                mem_props,
                name,
                driver,
                limits: props.limits,
                pool,
                cb,
                fence,
                dsl,
                layout,
                dpool,
                set,
                qpool,
                ts_period_ns: props.limits.timestamp_period as f64,
                pipes: Vec::new(),
                buffers: Vec::new(),
                dispatches: 0,
                submits: 0,
                gpu_seconds: 0.0,
                wait_seconds: 0.0,
                // Measured on an RX 6800 XT: a full barrier costs ~2 us per dispatch, a
                // compute-only one ~4 us (the driver handles it differently), so Full it is.
                barriers: Barriers::Full,
                spin: true,
                profile: None,
                ppool,
                plabels: Vec::new(),
            })
        }
    }

    fn mem_type(&self, bits: u32, want: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.mem_props.memory_type_count).find(|&i| {
            bits & (1 << i) != 0 && self.mem_props.memory_types[i as usize].property_flags.contains(want)
        })
    }

    /// A storage buffer of `bytes`: device-local, or host-visible and persistently mapped.
    pub fn buffer(&mut self, bytes: u64, host: bool) -> Res<Buffer> {
        unsafe {
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
            let buf = self
                .device
                .create_buffer(&vk::BufferCreateInfo::default().size(bytes.max(4)).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE), None)
                .map_err(vkerr("vkCreateBuffer"))?;
            let req = self.device.get_buffer_memory_requirements(buf);
            let idx = if host {
                let hv = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
                self.mem_type(req.memory_type_bits, hv | vk::MemoryPropertyFlags::HOST_CACHED)
                    .or_else(|| self.mem_type(req.memory_type_bits, hv))
            } else {
                self.mem_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            }
            .ok_or("no suitable memory type")?;
            let mem = self
                .device
                .allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(idx), None)
                .map_err(vkerr("vkAllocateMemory"))?;
            self.device.bind_buffer_memory(buf, mem, 0).map_err(vkerr("bind memory"))?;
            let ptr = if host {
                self.device.map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).map_err(vkerr("map"))? as *mut u8
            } else {
                std::ptr::null_mut()
            };
            self.buffers.push((buf, mem));
            Ok(Buffer { buf, mem, bytes, ptr })
        }
    }

    /// Binds `b` as the memory pool every kernel addresses (bindings 0 and 1).
    pub fn bind_pool(&mut self, b: &Buffer) {
        let info = [vk::DescriptorBufferInfo { buffer: b.buf, offset: 0, range: vk::WHOLE_SIZE }];
        let writes = [0u32, 1].map(|binding| {
            vk::WriteDescriptorSet::default()
                .dst_set(self.set)
                .dst_binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&info)
        });
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        let _ = b.mem;
    }

    pub fn kernel(&mut self, name: &'static str, spv: &[u8]) -> Res<Kernel> {
        unsafe {
            let words = ash::util::read_spv(&mut std::io::Cursor::new(spv)).map_err(|e| e.to_string())?;
            let module = self
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
                .map_err(vkerr("shader module"))?;
            let entry: &CStr = c"main";
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(entry);
            let ci = [vk::ComputePipelineCreateInfo::default().stage(stage).layout(self.layout)];
            let pipe = self
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &ci, None)
                .map_err(|(_, e)| format!("compute pipeline: {e:?}"))?[0];
            self.device.destroy_shader_module(module, None);
            self.pipes.push(pipe);
            Ok(Kernel { pipe, name })
        }
    }

    fn stamp(&mut self, label: &'static str) {
        if self.profile.is_some() && (self.plabels.len() as u32) < PROFILE_SLOTS {
            let i = self.plabels.len() as u32;
            unsafe { self.device.cmd_write_timestamp(self.cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, self.ppool, i) };
            self.plabels.push(label);
        }
    }

    pub fn begin(&mut self) {
        unsafe {
            let d = &self.device;
            d.reset_command_buffer(self.cb, vk::CommandBufferResetFlags::empty()).unwrap();
            d.begin_command_buffer(self.cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
                .unwrap();
            d.cmd_reset_query_pool(self.cb, self.qpool, 0, 2);
            d.cmd_write_timestamp(self.cb, vk::PipelineStageFlags::TOP_OF_PIPE, self.qpool, 0);
            d.cmd_bind_descriptor_sets(self.cb, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[self.set], &[]);
        }
        if self.profile.is_some() {
            unsafe { self.device.cmd_reset_query_pool(self.cb, self.ppool, 0, PROFILE_SLOTS) };
            self.plabels.clear();
            self.stamp("start");
        }
    }

    /// Compute and transfer writes made visible to compute and transfer reads/writes.
    fn barrier_full(&self) {
        let rw = vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE;
        let mb = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(rw)];
        let st = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
        unsafe { self.device.cmd_pipeline_barrier(self.cb, st, st, vk::DependencyFlags::empty(), &mb, &[], &[]) };
    }

    fn barrier_compute(&self) {
        let mb = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
        let st = vk::PipelineStageFlags::COMPUTE_SHADER;
        unsafe { self.device.cmd_pipeline_barrier(self.cb, st, st, vk::DependencyFlags::empty(), &mb, &[], &[]) };
    }

    fn barrier(&self) {
        match self.barriers {
            Barriers::Full => self.barrier_full(),
            Barriers::Compute => self.barrier_compute(),
            Barriers::Off => {}
        }
    }

    /// Records a dispatch followed by a full compute/transfer barrier.
    pub fn dispatch(&mut self, k: &Kernel, groups: [u32; 3], push: &[u32]) {
        assert!(push.len() <= PUSH_WORDS);
        let mut words = [0u32; PUSH_WORDS];
        words[..push.len()].copy_from_slice(push);
        unsafe {
            let bytes = std::slice::from_raw_parts(words.as_ptr() as *const u8, PUSH_WORDS * 4);
            self.device.cmd_bind_pipeline(self.cb, vk::PipelineBindPoint::COMPUTE, k.pipe);
            self.device.cmd_push_constants(self.cb, self.layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
            self.device.cmd_dispatch(self.cb, groups[0], groups[1], groups[2]);
        }
        self.dispatches += 1;
        self.barrier();
        self.stamp(k.name);
    }

    /// Records a dispatch of `n` threads of a 64-wide 1-D kernel (2-D grid when large).
    pub fn dispatch_1d(&mut self, k: &Kernel, n: usize, push: &[u32]) {
        if n == 0 {
            return;
        }
        let groups = n.div_ceil(64);
        let gx = groups.min(65535);
        let gy = groups.div_ceil(gx);
        self.dispatch(k, [gx as u32, gy as u32, 1], push);
    }

    pub fn copy(&mut self, src: &Buffer, src_word: usize, dst: &Buffer, dst_word: usize, words: usize) {
        if words == 0 {
            return;
        }
        let region = [vk::BufferCopy { src_offset: (src_word * 4) as u64, dst_offset: (dst_word * 4) as u64, size: (words * 4) as u64 }];
        self.barrier_full();
        unsafe { self.device.cmd_copy_buffer(self.cb, src.buf, dst.buf, &region) };
        self.barrier_full();
        self.stamp("copy");
    }

    /// Submits an empty command buffer and waits: the fixed round-trip cost.
    pub fn roundtrip(&mut self) -> f64 {
        let t0 = std::time::Instant::now();
        self.begin();
        self.submit();
        t0.elapsed().as_secs_f64()
    }

    /// Ends, submits and waits; returns the GPU time of the command buffer in seconds.
    pub fn submit(&mut self) -> f64 {
        let t0 = std::time::Instant::now();
        unsafe {
            let d = &self.device;
            let host = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)];
            d.cmd_pipeline_barrier(
                self.cb,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &host,
                &[],
                &[],
            );
            d.cmd_write_timestamp(self.cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, self.qpool, 1);
            d.end_command_buffer(self.cb).unwrap();
            let cbs = [self.cb];
            d.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cbs)], self.fence).unwrap();
            if self.spin {
                while !d.get_fence_status(self.fence).unwrap() {
                    std::hint::spin_loop();
                }
            } else {
                d.wait_for_fences(&[self.fence], true, u64::MAX).unwrap();
            }
            d.reset_fences(&[self.fence]).unwrap();
            let mut ts = [0u64; 2];
            d.get_query_pool_results(self.qpool, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)
                .unwrap();
            let gpu = ts[1].wrapping_sub(ts[0]) as f64 * self.ts_period_ns * 1e-9;
            if self.profile.is_some() && self.plabels.len() > 1 {
                let n = self.plabels.len();
                let mut v = vec![0u64; n];
                d.get_query_pool_results(self.ppool, 0, &mut v, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)
                    .unwrap();
                let period = self.ts_period_ns;
                let prof = self.profile.as_mut().unwrap();
                for i in 1..n {
                    let e = prof.entry(self.plabels[i]).or_insert((0.0, 0));
                    e.0 += v[i].wrapping_sub(v[i - 1]) as f64 * period * 1e-9;
                    e.1 += 1;
                }
            }
            self.submits += 1;
            self.gpu_seconds += gpu;
            self.wait_seconds += t0.elapsed().as_secs_f64();
            gpu
        }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            let d = &self.device;
            let _ = d.device_wait_idle();
            for &p in &self.pipes {
                d.destroy_pipeline(p, None);
            }
            for &(b, m) in &self.buffers {
                d.destroy_buffer(b, None);
                d.free_memory(m, None);
            }
            d.destroy_query_pool(self.qpool, None);
            d.destroy_query_pool(self.ppool, None);
            d.destroy_descriptor_pool(self.dpool, None);
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_descriptor_set_layout(self.dsl, None);
            d.destroy_fence(self.fence, None);
            d.destroy_command_pool(self.pool, None);
            d.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
