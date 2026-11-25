// SPDX-License-Identifier: GPL-3.0-only

use taudem_rs::{
    DIRECTION_NODATA as ND, FEL_NODATA, Grid, PipelineOptions, Retain, SCALAR_NODATA, area_d8,
    d8_flow, pipeline, pit_remove,
};

#[test]
fn nested_depressions_fill_to_the_external_sill() {
    let grid = Grid::new(7, 7).unwrap();
    let mut z = vec![
        9., 9., 9., 9., 9., 9., 9., 9., 6., 6., 6., 6., 6., 9., 9., 6., 3., 3., 3., 6., 9., 5., 4.,
        3., 1., 3., 6., 9., 9., 6., 3., 3., 3., 6., 9., 9., 6., 6., 6., 6., 6., 9., 9., 9., 9., 9.,
        9., 9., 9.,
    ];
    pit_remove(grid, &mut z, -9999., false, None);
    assert_eq!(&z[21..28], &[5., 5., 5., 5., 5., 6., 9.]);
    assert_eq!(z[2 * 7 + 3], 5.);
    assert_eq!(z[1], 9.);
}

#[test]
fn four_way_connectivity_excludes_diagonal_exits() {
    let grid = Grid::new(3, 3).unwrap();
    let z = [1., 8., 8., 8., 0., 8., 8., 8., 8.];
    let mut d8 = z;
    let mut d4 = z;
    pit_remove(grid, &mut d8, -9999., false, None);
    pit_remove(grid, &mut d4, -9999., true, None);
    assert_eq!(d8[4], 1.);
    assert_eq!(d4[4], 8.);
}

#[test]
fn nodata_and_mask_cells_provide_internal_outlets() {
    let grid = Grid::new(5, 5).unwrap();
    let mut z = [9.; 25];
    z[12] = -9999.;
    z[6] = 2.;
    z[7] = 3.;
    z[18] = 4.;
    let mut mask = [0; 25];
    mask[18] = 1;
    pit_remove(grid, &mut z, -9999., true, Some(&mask));
    assert_eq!(z[12], FEL_NODATA);
    assert_eq!(z[6], 3.);
    assert_eq!(z[7], 3.);
    assert_eq!(z[18], 4.);
    let flow = d8_flow(grid, &z, FEL_NODATA, &[(1., 1.)], true);
    assert!(flow.direction.iter().all(|&code| code == ND));
    assert!(flow.slope.unwrap().iter().all(|&s| s == SCALAR_NODATA));
}

#[test]
fn flats_route_towards_an_outlet_without_artificial_slope() {
    let grid = Grid::new(7, 5).unwrap();
    let mut z = Vec::new();
    for _ in 0..5 {
        z.extend([9., 9., 5., 5., 5., 4., 3.]);
    }
    let result = pipeline(
        grid,
        z.clone(),
        PipelineOptions {
            nodata: -9999.,
            four_way: false,
            mask: None,
            spacing: &[(1., 1.)],
            check_contamination: false,
            retain: Retain {
                filled: true,
                direction: true,
                slope: true,
            },
        },
    );
    assert_eq!(result.filled.unwrap(), z);
    let p = result.direction.unwrap();
    let sd8 = result.slope.unwrap();
    assert_eq!(&p[15..20], &[1, 1, 1, 1, 1]);
    assert_eq!(&sd8[15..20], &[4., 0., 0., 1., 1.]);
    assert_eq!(&result.area[15..20], &[1., 2., 3., 4., 5.]);
    assert_eq!(p[14], ND);
}

#[test]
fn slope_uses_rectangular_distance_and_cardinal_ties() {
    let grid = Grid::new(3, 3).unwrap();
    let z = [10., 8., 10., 10., 10., 6., 10., 10., 10.];
    let f = d8_flow(grid, &z, -9999., &[(2., 1.)], true);
    assert_eq!(f.direction[4], 1);
    assert_eq!(f.slope.unwrap()[4], 2.);
    assert_eq!(
        d8_flow(grid, &z, -9999., &[(3., 1.)], false).direction[4],
        3
    );
}

#[test]
fn area_propagates_edge_and_nodata_contamination_downstream() {
    let grid = Grid::new(5, 5).unwrap();
    let mut p = [1; 25];
    p[6] = ND;
    // The second interior row receives a contaminated boundary/hole neighbor.
    // The final interior row is downstream of the hole in the middle row.
    p[12] = 7;
    let checked = area_d8(grid, &p, true);
    let unchecked = area_d8(grid, &p, false);
    assert_eq!(checked[12], SCALAR_NODATA);
    assert_eq!(checked[17], SCALAR_NODATA);
    assert_eq!(unchecked[6], SCALAR_NODATA);
    assert_eq!(unchecked[12], 3.);
    assert_eq!(unchecked[17], 6.);
    assert_eq!(unchecked[19], 8.);
}

#[test]
fn area_leaves_cycles_unresolved_and_handles_a_single_cell() {
    let grid = Grid::new(2, 1).unwrap();
    assert_eq!(area_d8(grid, &[1, 5], false), vec![SCALAR_NODATA; 2]);
    let grid = Grid::new(1, 1).unwrap();
    assert_eq!(area_d8(grid, &[1], false), vec![1.]);
    assert_eq!(area_d8(grid, &[1], true), vec![SCALAR_NODATA]);
}
