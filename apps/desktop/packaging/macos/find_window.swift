// Prints the window number of the largest on-screen window owned by an app, for
// `screencapture -l`. Used by screenshots.sh.
//
//   swiftc find_window.swift -o find_window && ./find_window PiedPiper
import CoreGraphics
import Foundation

let owner = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "PiedPiper"
let windows = (CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID)
    as? [[String: Any]]) ?? []
var best: (id: Int, area: Double)?
for w in windows {
    guard (w[kCGWindowOwnerName as String] as? String) == owner,
          let id = w[kCGWindowNumber as String] as? Int,
          let bounds = w[kCGWindowBounds as String] as? [String: Any] else { continue }
    let width = (bounds["Width"] as? NSNumber)?.doubleValue ?? 0
    let height = (bounds["Height"] as? NSNumber)?.doubleValue ?? 0
    if width * height > 10_000, best == nil || width * height > best!.area {
        best = (id, width * height)
    }
}
guard let found = best else { exit(1) }
print(found.id)
