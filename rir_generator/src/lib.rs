#[macro_use]
extern crate itertools;

use ndarray;
use std::f64::consts::PI;
use std::f64::EPSILON;

#[derive(Debug, Clone)]
pub enum MicrophoneType {
    Bidirectional,
    Hypercardioid,
    Cardioid,
    Subcardioid,
    Omnidirectional,
}

impl Into<f64> for MicrophoneType {
    fn into(self) -> f64 {
        match self {
            Self::Bidirectional => 0.0,
            Self::Hypercardioid => 0.25,
            Self::Cardioid => 0.5,
            Self::Subcardioid => 0.75,
            Self::Omnidirectional => 1.0,
        }
    }
}

pub struct InvalidMicrophoneCharError {}

impl TryFrom<char> for MicrophoneType {
    type Error = InvalidMicrophoneCharError;

    fn try_from(x: char) -> Result<Self, Self::Error> {
        match x {
            'b' => Ok(Self::Bidirectional),
            'h' => Ok(Self::Hypercardioid),
            'c' => Ok(Self::Cardioid),
            's' => Ok(Self::Subcardioid),
            'o' => Ok(Self::Omnidirectional),
            _ => Err(InvalidMicrophoneCharError {}),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Position {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}
pub type Room = Position;

impl From<&[f64; 3]> for Position {
    fn from(x: &[f64; 3]) -> Position {
        Position {
            x: x[0],
            y: x[1],
            z: x[2],
        }
    }
}

impl From<Vec<f64>> for Position {
    fn from(x: Vec<f64>) -> Position {
        Position {
            x: x[0],
            y: x[1],
            z: x[2],
        }
    }
}

#[derive(Debug, Clone)]
pub struct Betas {
    pub x: [f64; 2],
    pub y: [f64; 2],
    pub z: [f64; 2],
}

impl From<f64> for Betas {
    fn from(x: f64) -> Betas {
        Betas {
            x: [x, x],
            y: [x, x],
            z: [x, x],
        }
    }
}

impl From<&[f64; 6]> for Betas {
    fn from(x: &[f64; 6]) -> Betas {
        Betas {
            x: [x[0], x[1]],
            y: [x[2], x[3]],
            z: [x[4], x[5]],
        }
    }
}

impl From<Vec<f64>> for Betas {
    fn from(x: Vec<f64>) -> Betas {
        Betas {
            x: [x[0], x[1]],
            y: [x[2], x[3]],
            z: [x[4], x[5]],
        }
    }
}

#[derive(Debug, Clone)]
pub struct Angle {
    pub phi: f64,
    pub theta: f64,
}

impl From<f64> for Angle {
    fn from(x: f64) -> Angle {
        Angle { phi: x, theta: x }
    }
}

impl From<&[f64; 2]> for Angle {
    fn from(x: &[f64; 2]) -> Angle {
        Angle {
            phi: x[0],
            theta: x[1],
        }
    }
}

impl From<Vec<f64>> for Angle {
    fn from(x: Vec<f64>) -> Angle {
        Angle {
            phi: x[0],
            theta: x[1],
        }
    }
}

#[derive(Debug, Clone)]
pub struct Receiver {
    pub position: Position,
    pub microphone_type: MicrophoneType,
    pub angle: Angle,
}

trait FloatSinc {
    fn sinc(self) -> f64;
}

impl FloatSinc for f64 {
    fn sinc(self) -> f64 {
        let eps = EPSILON.copysign(self);
        (self + eps).sin() / (self + eps)
    }
}

fn sim_microphone(p: &Position, a: &Angle, t: &MicrophoneType) -> f64 {
    match t {
        MicrophoneType::Hypercardioid => 1.0,
        _ => {
            let rho: f64 = t.clone().into();
            let vartheta = (p.z / (p.x.powi(2) + p.y.powi(2) + p.z.powi(2)).sqrt()).acos();
            let varphi = p.y.atan2(p.x);

            let gain = (PI / 2.0 - a.theta).sin() * (vartheta).sin() * (a.phi - varphi).cos()
                + (PI / 2.0 - a.theta).cos() * (vartheta).cos();
            rho + (1.0 - rho) * gain
        }
    }
}

pub fn compute_rir(
    c: f64,
    fs: f64,
    receivers: &[Receiver],
    source: &Position,
    room: &Room,
    beta: &Betas,
    n_samples: usize,
    n_order: i64,
    enable_highpass_filter: bool,
) -> ndarray::Array2<f64> {
    // Temporary variables and constants (image-method)
    let fc = 0.5; // The normalized cut-off frequency equals (fs/2) / fs = 0.5
    let tw = (2.0 * (0.004 * fs).round()) as usize; // The width of the low-pass FIR equals 8 ms
    let cts = c / fs;
    let mut imp = ndarray::Array2::<f64>::zeros((receivers.len(), n_samples));

    let source = Position {
        x: source.x / cts,
        y: source.y / cts,
        z: source.z / cts,
    };
    let room = Room {
        x: room.x / cts,
        y: room.y / cts,
        z: room.z / cts,
    };

    for (
        Receiver {
            position: receiver,
            microphone_type: mtype,
            angle,
        },
        mut imp,
    ) in receivers.iter().zip(imp.axis_iter_mut(ndarray::Axis(0)))
    {
        let receiver = Position {
            x: receiver.x / cts,
            y: receiver.y / cts,
            z: receiver.z / cts,
        };

        let n1 = (n_samples as f64 / (2.0 * room.x)).ceil() as i64;
        let n2 = (n_samples as f64 / (2.0 * room.y)).ceil() as i64;
        let n3 = (n_samples as f64 / (2.0 * room.z)).ceil() as i64;

        // Generate room impulse response
        for (mx, my, mz, q, j, k) in iproduct!(-n1..=n1, -n2..=n2, -n3..=n3, 0..=1, 0..=1, 0..=1) {
            let rm = Room {
                x: 2.0 * mx as f64 * room.x,
                y: 2.0 * my as f64 * room.y,
                z: 2.0 * mz as f64 * room.z,
            };

            let rp_plus_rm = Position {
                x: (1 - 2 * q) as f64 * source.x - receiver.x + rm.x,
                y: (1 - 2 * j) as f64 * source.y - receiver.y + rm.y,
                z: (1 - 2 * k) as f64 * source.z - receiver.z + rm.z,
            };
            let refl = [
                beta.x[0].powi((mx - q).abs() as i32) * beta.x[1].powi((mx).abs() as i32),
                beta.y[0].powi((my - j).abs() as i32) * beta.y[1].powi((my).abs() as i32),
                beta.z[0].powi((mz - k).abs() as i32) * beta.z[1].powi((mz).abs() as i32),
            ];

            let dist = (rp_plus_rm.x.powi(2) + rp_plus_rm.y.powi(2) + rp_plus_rm.z.powi(2)).sqrt();
            let fdist = (dist).floor();

            if (fdist as usize) < n_samples
                && (n_order == -1
                    || (2 * mx - q).abs() + (2 * my - j).abs() + (2 * mz - k).abs() <= n_order)
            {
                let gain =
                    sim_microphone(&rp_plus_rm, &angle, &mtype) * refl[0] * refl[1] * refl[2]
                        / (4.0 * PI * dist * cts);

                let mut lpi = ndarray::Array::zeros(tw);

                for (n, lpi) in lpi.iter_mut().enumerate() {
                    let t = ((n as f64) - 0.5 * (tw as f64) + 1.0) - (dist - fdist);
                    *lpi = 0.5
                        * (1.0 + (2.0 * PI * t / (tw as f64)).cos())
                        * 2.0
                        * fc
                        * (2.0 * PI * fc * t).sinc();
                }

                let start_position = (fdist - (tw as f64 / 2.0) + 1.0) as usize;

                for (imp, lpi) in imp
                    .slice_mut(ndarray::s![start_position..])
                    .iter_mut()
                    .zip(lpi)
                {
                    *imp += gain * lpi
                }
            }
        }
    }

    if enable_highpass_filter {
        // 'Original' high-pass filter as proposed by Allen and Berkley.
        // Temporary variables and constants (high-pass filter)
        let w = 2.0 * PI * 100.0 / fs; // The cut-off frequency equals 100 Hz
        let r1 = -w.exp();
        let b1 = 2.0 * r1 * w.cos();
        let b2 = -r1 * r1;
        let a1 = -(1.0 + r1);

        for mut imp in imp.axis_iter_mut(ndarray::Axis(0)) {
            let mut y = [0.0; 3];

            for x0 in imp.iter_mut() {
                y[2] = y[1];
                y[1] = y[0];
                y[0] = b1 * y[1] + b2 * y[2] + *x0;
                *x0 = y[0] + a1 * y[1] + r1 * y[2];
            }
        }
    }

    imp
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_example() {
        let imp = compute_rir(
            340.0,
            16000.0,
            &vec![Receiver {
                position: Position::from(&[2.0, 1.5, 2.0]),
                microphone_type: MicrophoneType::Omnidirectional,
                angle: Angle::from(0.0),
            }],
            &Position::from(&[2.0, 3.5, 2.0]),
            &Room::from(&[5.0, 4.0, 6.0]),
            &Betas::from(0.4),
            4096,
            -1,
            true,
        );

        assert!(imp.into_iter().any(|x| x > 0.0))
    }
}
