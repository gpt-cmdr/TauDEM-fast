// SPDX-License-Identifier: GPL-3.0-only

use taudem_rs::profile;

#[test]
fn spans_nest_and_render_as_json() {
    {
        let _outer = profile::span("outer");
        profile::count("things", 2);
        profile::count("things", 3);
        let _inner = profile::span("inner \"quoted\"");
        profile::buffer("scratch", 64);
    }
    let report = profile::report();
    let outer = report.spans.iter().find(|s| s.name == "outer").unwrap();
    let inner = report
        .spans
        .iter()
        .find(|s| s.name.starts_with("inner"))
        .unwrap();
    assert_eq!(inner.depth, outer.depth + 1);
    assert!(inner.start_s >= outer.start_s);
    assert!(report.counters.contains(&("things", 5)));
    let json = report.to_json(&[("cells", "8".into())], 8);
    assert!(json.contains("\"cells\": 8,"));
    assert!(json.contains("inner \\\"quoted\\\""));
    assert!(json.contains("\"bytes_per_cell\": 8.000"));
    assert_eq!(json.matches('{').count(), json.matches('}').count());
}
