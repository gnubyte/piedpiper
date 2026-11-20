// Test harness: runs a Finder service on files the way the Finder does, by putting their URLs on
// a pasteboard and calling NSPerformService. macOS starts PiedPiper.app if needed.
//
//   swift perform_service.swift "Compress with PiedPiper" /path/a.txt /path/folder
import AppKit

let args = CommandLine.arguments
guard args.count >= 3 else {
    FileHandle.standardError.write("usage: perform_service.swift SERVICE PATH...\n".data(using: .utf8)!)
    exit(2)
}
let pasteboard = NSPasteboard(name: NSPasteboard.Name("PiedPiperServiceTest"))
pasteboard.clearContents()
let urls = args.dropFirst(2).map { URL(fileURLWithPath: $0) as NSURL }
guard pasteboard.writeObjects(urls) else {
    FileHandle.standardError.write("could not write the file URLs to the pasteboard\n".data(using: .utf8)!)
    exit(1)
}
let ok = NSPerformService(args[1], pasteboard)
print("NSPerformService(\(args[1])): \(ok)")
exit(ok ? 0 : 1)
