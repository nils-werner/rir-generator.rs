use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rir_generator::{self, InvalidMicrophoneCharError};

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn compute_rir(
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
) -> Result<&PyArray2<f64>, MyInvalidMicrophoneCharError> {
    let receiver = rir_generator::Receiver {
        position: rir_generator::Position::from(receiver),
        microphone_type: rir_generator::MicrophoneType::try_from(microphone)?,
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
        n_order,
        enable_highpass_filter,
    )
    .into_pyarray(_py))
}

/// A Python module implemented in Rust.
#[pymodule]
fn rir_generator_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(compute_rir, m)?)?;
    Ok(())
}
struct MyInvalidMicrophoneCharError(InvalidMicrophoneCharError);

impl From<MyInvalidMicrophoneCharError> for PyErr {
    fn from(_: MyInvalidMicrophoneCharError) -> Self {
        PyValueError::new_err("Incorrect microphone character")
    }
}

impl From<InvalidMicrophoneCharError> for MyInvalidMicrophoneCharError {
    fn from(other: InvalidMicrophoneCharError) -> Self {
        Self(other)
    }
}
