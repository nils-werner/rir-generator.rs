use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn generate(
    _py: Python,
    c: f64,
    fs: f64,
    receiver: Vec<f64>,
    microphone: char,
    angle: Vec<f64>,
    source: Vec<f64>,
    room: Vec<f64>,
    beta: Vec<f64>,
    n_samples: usize,
    n_order: i64,
    enable_highpass_filter: bool,
) -> PyResult<&PyArray2<f64>> {
    let receiver = rir_generator::Receiver {
        position: rir_generator::Position::from(receiver),
        microphone_type: rir_generator::MicrophoneType::try_from(microphone)
            .map_err(|error| PyValueError::new_err(error.to_string()))?,
        angle: rir_generator::Angle::from(angle),
    };

    Ok(rir_generator::compute_rir(
        c,
        fs,
        &[receiver],
        &rir_generator::Position::from(source),
        &rir_generator::Room::from(room),
        &rir_generator::Betas::from(beta),
        n_samples,
        rir_generator::FilterOrder::from(n_order),
        enable_highpass_filter,
    )
    .into_pyarray(_py))
}

/// A Python module implemented in Rust.
#[pymodule]
fn rir_generator_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate, m)?)?;
    Ok(())
}
