use crate::rir::{compute_rir, Angle, Betas, Microphone, Position, Room};
use numpy::ndarray::{Array2, Axis};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
use pyo3::{pyfunction, pymodule, Py, types::PyModule, PyResult, Python};

#[pymodule]
fn rir_generator(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    #[pyfunction]
    fn generate(
        c: f64,
        fs: f64,
        receivers: PyReadonlyArray2<'_, f64>,
        source: PyReadonlyArray1<'_, f64>,
        room: PyReadonlyArray1<'_, f64>,
        beta: PyReadonlyArray2<'_, f64>,
        microphone_types: Vec<char>,
        microphone_angles: PyReadonlyArray2<'_, f64>,
        n_samples: usize,
        n_order: i64,
        enable_highpass_filter: bool,
    ) -> Py<PyArray2<f64>> {
        let receivers: Vec<Position> = receivers
            .as_array()
            .outer_iter()
            .map(|row| Position::try_from(row.as_slice().unwrap()).unwrap())
            .collect();
        let source = Position::from(source.as_slice().unwrap().try_into().unwrap());
        let room = Room::try_from(room.as_slice().unwrap()).unwrap();
        let beta = Betas::from(beta.as_array().as_slice().try_into().unwrap());
        let microphone_types: Vec<Microphone> = microphone_types
            .iter()
            .map(|row| Microphone::try_from(row.clone().try_into().unwrap()).unwrap())
            .collect();
        let microphone_angles: Vec<Angle> = microphone_angles
            .as_array()
            .outer_iter()
            .map(|row| Angle::try_from(row.as_slice().unwrap().try_into().unwrap()).unwrap())
            .collect();

        let mut arr = Array2::zeros((receivers.len(), n_samples));

        let intermediate = compute_rir(
            c,
            fs,
            &receivers,
            &source,
            &room,
            &beta,
            &microphone_types,
            &microphone_angles,
            n_samples,
            n_order,
            enable_highpass_filter,
        );

        for (i, mut row) in arr.axis_iter_mut(Axis(0)).enumerate() {
            for (j, col) in row.iter_mut().enumerate() {
                *col = intermediate[i][j];
            }
        }

        todo!()
    }

    Ok(())
}
