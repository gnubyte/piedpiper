//! Minimal Vulkan compute runtime: one device and queue, one storage-buffer memory pool bound
//! to every kernel (bindings 0 and 1 alias it as `float[]` and `uint[]`), 32 words of push
//! constants per dispatch, and one command buffer that is recorded, submitted and waited on.
//!
//! The Vulkan loader is opened at run time (`vulkan-1.dll`, `libvulkan.so.1`), so a build with
//! the `gpu` feature still starts on machines without a Vulkan driver; it just reports that no
//! GPU is available.

use ash::vk;

pub const PUSH_WORDS: usize = 32;

pub type Res<T> = std::result::Result<T, String>;

/// Testing aid: `PIEDPIPER_GPU_FAIL_AFTER=N` makes every submission after the first N fail, as a
/// lost device would, to exercise the fallback to the CPU.
fn injected_failure() -> bool {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;
    static AFTER: OnceLock<Option<u64>> = OnceLock::new();
    static COUNT: AtomicU64 = AtomicU64::new(0);
    match AFTER.get_or_init(|| std::env::var("PIEDPIPER_GPU_FAIL_AFTER").ok().and_then(|v| v.parse().ok())) {
        Some(n) => COUNT.fetch_add(1, Ordering::Relaxed) >= *n,
        None => false,
    }
}

fn vkerr(what: &str) -> impl Fn(vk::Result) -> String + '_ {
    move |e| format!("{what}: {e:?}")
}

pub struct Buffer {
    pub buf: vk::Buffer,
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
        // SAFETY: as above.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), (self.host() as *mut f32).add(word), data.len()) }
    }

    pub fn f32s(&self, word: usize, len: usize) -> &[f32] {
        assert!(((word + len) * 4) as u64 <= self.bytes);
        // SAFETY: as above; the slice lives as long as the buffer.
        unsafe { std::slice::from_raw_parts((self.host() as *const f32).add(word), len) }
    }
}

pub struct Kernel {
    pipe: vk::Pipeline,
}

/// Device, queue, command buffer and descriptor set.
pub struct Gpu {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    pub name: String,
    pub driver: String,
    pool: vk::CommandPool,
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    dsl: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    dpool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    pipes: Vec<vk::Pipeline>,
    buffers: Vec<(vk::Buffer, vk::DeviceMemory)>,
}

// SAFETY: a `Gpu` (and the buffers it owns) is used by one thread at a time: a model owns it
// while it runs, and the idle pool hands it over under a mutex.
unsafe impl Send for Gpu {}
unsafe impl Send for Buffer {}

impl Gpu {
    /// Opens the first discrete GPU (or `PIEDPIPER_GPU=<index>`) and checks the float controls
    /// the bit-exact kernels need.
    pub fn new() -> Res<Gpu> {
        unsafe {
            let entry = ash::Entry::load().map_err(|e| format!("no Vulkan driver ({e})"))?;
            let app = vk::ApplicationInfo::default().application_name(c"PiedPiper").api_version(vk::API_VERSION_1_2);
            let instance = entry
                .create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)
                .map_err(vkerr("vkCreateInstance"))?;
            match Self::open(entry, instance.clone()) {
                Ok(g) => Ok(g),
                Err(e) => {
                    instance.destroy_instance(None);
                    Err(e)
                }
            }
        }
    }

    unsafe fn open(entry: ash::Entry, instance: ash::Instance) -> Res<Gpu> {
        let pdevs = instance.enumerate_physical_devices().map_err(vkerr("vkEnumeratePhysicalDevices"))?;
        if pdevs.is_empty() {
            return Err("no Vulkan device".into());
        }
        let pick = match std::env::var("PIEDPIPER_GPU").ok().and_then(|s| s.parse::<usize>().ok()) {
            Some(i) => *pdevs.get(i).ok_or("PIEDPIPER_GPU names no device")?,
            None => *pdevs
                .iter()
                .find(|&&p| instance.get_physical_device_properties(p).device_type == vk::PhysicalDeviceType::DISCRETE_GPU)
                .unwrap_or(&pdevs[0]),
        };
        let base = instance.get_physical_device_properties(pick);
        if base.api_version < vk::API_VERSION_1_2 {
            return Err("the GPU's driver offers Vulkan below 1.2".into());
        }
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
        )
        .trim()
        .to_string();
        let need = [
            ("shaderDenormPreserveFloat32", v12.shader_denorm_preserve_float32),
            ("shaderRoundingModeRTEFloat32", v12.shader_rounding_mode_rte_float32),
            ("shaderSignedZeroInfNanPreserveFloat32", v12.shader_signed_zero_inf_nan_preserve_float32),
        ];
        for (n, v) in need {
            if v != vk::TRUE {
                return Err(format!("{name} does not support {n}, which bit-exact results need"));
            }
        }
        if props.limits.max_push_constants_size < (PUSH_WORDS * 4) as u32 || props.limits.max_compute_shared_memory_size < 32768 {
            return Err(format!("{name} has too few push constants or too little shared memory"));
        }
        let qfams = instance.get_physical_device_queue_family_properties(pick);
        let qf = qfams
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or("no compute queue")? as u32;
        let prio = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default().queue_family_index(qf).queue_priorities(&prio)];
        let device = instance
            .create_device(pick, &vk::DeviceCreateInfo::default().queue_create_infos(&qci), None)
            .map_err(vkerr("vkCreateDevice"))?;
        let queue = device.get_device_queue(qf, 0);
        let mem_props = instance.get_physical_device_memory_properties(pick);
        let pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default().flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER).queue_family_index(qf),
                None,
            )
            .map_err(vkerr("command pool"))?;
        let cb = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1),
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
            .map_err(vkerr("descriptor set layout"))?;
        let ranges = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size((PUSH_WORDS * 4) as u32)];
        let dsls = [dsl];
        let layout = device
            .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&dsls).push_constant_ranges(&ranges), None)
            .map_err(vkerr("pipeline layout"))?;
        let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 2 }];
        let dpool = device
            .create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None)
            .map_err(vkerr("descriptor pool"))?;
        let set = device
            .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(dpool).set_layouts(&dsls))
            .map_err(vkerr("descriptor set"))?[0];
        Ok(Gpu {
            _entry: entry,
            instance,
            device,
            queue,
            mem_props,
            name,
            driver,
            pool,
            cb,
            fence,
            dsl,
            layout,
            dpool,
            set,
            pipes: Vec::new(),
            buffers: Vec::new(),
        })
    }

    fn mem_type(&self, bits: u32, want: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.mem_props.memory_type_count)
            .find(|&i| bits & (1 << i) != 0 && self.mem_props.memory_types[i as usize].property_flags.contains(want))
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
            };
            let Some(idx) = idx else {
                self.device.destroy_buffer(buf, None);
                return Err("no suitable GPU memory type".into());
            };
            let mem = match self
                .device
                .allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(idx), None)
            {
                Ok(m) => m,
                Err(e) => {
                    self.device.destroy_buffer(buf, None);
                    return Err(format!("GPU memory ({} MB): {e:?}", req.size >> 20));
                }
            };
            self.buffers.push((buf, mem));
            self.device.bind_buffer_memory(buf, mem, 0).map_err(vkerr("bind memory"))?;
            let ptr = if host {
                self.device.map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).map_err(vkerr("map memory"))? as *mut u8
            } else {
                std::ptr::null_mut()
            };
            Ok(Buffer { buf, bytes, ptr })
        }
    }

    /// Frees every buffer (the device and the compiled kernels stay for the next model).
    pub fn free_buffers(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            for (b, m) in self.buffers.drain(..) {
                self.device.destroy_buffer(b, None);
                self.device.free_memory(m, None);
            }
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
    }

    pub fn kernel(&mut self, spv: &[u8]) -> Res<Kernel> {
        unsafe {
            let words = ash::util::read_spv(&mut std::io::Cursor::new(spv)).map_err(|e| e.to_string())?;
            let module = self
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
                .map_err(vkerr("shader module"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            let ci = [vk::ComputePipelineCreateInfo::default().stage(stage).layout(self.layout)];
            let made = self.device.create_compute_pipelines(vk::PipelineCache::null(), &ci, None);
            self.device.destroy_shader_module(module, None);
            let pipe = made.map_err(|(_, e)| format!("compute pipeline: {e:?}"))?[0];
            self.pipes.push(pipe);
            Ok(Kernel { pipe })
        }
    }

    pub fn begin(&mut self) -> Res<()> {
        unsafe {
            let d = &self.device;
            d.reset_command_buffer(self.cb, vk::CommandBufferResetFlags::empty()).map_err(vkerr("reset command buffer"))?;
            d.begin_command_buffer(self.cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
                .map_err(vkerr("begin command buffer"))?;
            d.cmd_bind_descriptor_sets(self.cb, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[self.set], &[]);
        }
        Ok(())
    }

    /// Compute and transfer writes made visible to compute and transfer reads and writes. (On
    /// an RX 6800 XT this costs about 2 us per dispatch; a compute-only barrier cost 4 us.)
    fn barrier(&self) {
        let rw = vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE;
        let mb = [vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(rw)];
        let st = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
        unsafe { self.device.cmd_pipeline_barrier(self.cb, st, st, vk::DependencyFlags::empty(), &mb, &[], &[]) };
    }

    /// Records a dispatch followed by a barrier.
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
        self.barrier();
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
        self.barrier();
        unsafe { self.device.cmd_copy_buffer(self.cb, src.buf, dst.buf, &region) };
        self.barrier();
    }

    /// Ends, submits and waits (polling the fence: the decoder is latency-bound, and a sleeping
    /// wait added a quarter millisecond per round trip on Windows). A lost device, or a GPU that
    /// has not finished after 30 s, is an error; the caller then stops using the GPU.
    pub fn submit(&mut self) -> Res<()> {
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
            d.end_command_buffer(self.cb).map_err(vkerr("end command buffer"))?;
            if injected_failure() {
                return Err("injected failure (PIEDPIPER_GPU_FAIL_AFTER)".into());
            }
            let cbs = [self.cb];
            d.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cbs)], self.fence)
                .map_err(vkerr("vkQueueSubmit"))?;
            let t0 = std::time::Instant::now();
            let mut spins = 0u32;
            loop {
                match d.get_fence_status(self.fence) {
                    Ok(true) => break,
                    Ok(false) => {}
                    Err(e) => return Err(format!("the GPU stopped responding ({e:?})")),
                }
                spins = spins.wrapping_add(1);
                if spins.is_multiple_of(4096) && t0.elapsed().as_secs() >= 30 {
                    return Err("the GPU did not finish within 30 s".into());
                }
                std::hint::spin_loop();
            }
            d.reset_fences(&[self.fence]).map_err(vkerr("reset fence"))?;
        }
        Ok(())
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        self.free_buffers();
        unsafe {
            let d = &self.device;
            for &p in &self.pipes {
                d.destroy_pipeline(p, None);
            }
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
