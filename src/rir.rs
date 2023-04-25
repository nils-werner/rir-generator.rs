use std::f64::consts::PI;
use std::f64::EPSILON;
use std::ops::Div;

#[derive(Debug, Clone)]
pub enum MicrophoneType {
    Bidirectional,
    Hypercardioid,
    Cardioid,
    Subcardioid,
    Omnidirectional,
}

impl MicrophoneType {
    fn adjust_gain(&self, gain: f64) -> f64 {
        let rho = match self {
            Self::Bidirectional => 0.0,
            Self::Hypercardioid => 0.25,
            Self::Cardioid => 0.5,
            Self::Subcardioid => 0.75,
            Self::Omnidirectional => 1.0,
        };

        rho + (1.0 - rho) * gain
    }
}

impl TryFrom<char> for MicrophoneType {
    type Error = &'static str;

    fn try_from(x: char) -> Result<Self, Self::Error> {
        match x {
            'b' => Ok(Self::Bidirectional),
            'h' => Ok(Self::Hypercardioid),
            'c' => Ok(Self::Cardioid),
            's' => Ok(Self::Subcardioid),
            'o' => Ok(Self::Omnidirectional),
            _ => Err("Invalid character given"),
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

impl Position {
    pub fn radius(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }
}

impl Div<f64> for &Position {
    // The division of rational numbers is a closed operation.
    type Output = Position;

    fn div(self, rhs: f64) -> Self::Output {
        Position {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
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

fn sim_microphone(p: Position, recv: &Receiver) -> f64 {
    match recv.microphone_type {
        MicrophoneType::Hypercardioid => 1.0,
        _ => {
            let (vartheta, varphi) = ((p.z / p.radius()).acos(), p.y.atan2(p.x));
            let Angle { theta, phi } = recv.angle;

            let gain = (PI / 2.0 - theta).sin() * vartheta.sin() * (phi - varphi).cos()
                + (PI / 2.0 - theta).cos() * vartheta.cos();

            recv.microphone_type.adjust_gain(gain)
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
) -> Vec<Vec<f64>> {
    const FC: f64 = 0.5;
    // Temporary variables and constants (image-method)
    let tw = (2.0 * (0.004 * fs).round()) as usize; // The width of the low-pass FIR equals 8 ms
    let cts = c / fs;
    let mut imp = vec![vec![0.0; n_samples]; receivers.len()];

    let (source, room) = (source / cts, room / cts);

    for (receiver, imp) in receivers.iter().zip(imp.iter_mut()) {
        let scaled_receiver = &receiver.position / cts;

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
                x: (1 - 2 * q) as f64 * source.x - scaled_receiver.x + rm.x,
                y: (1 - 2 * j) as f64 * source.y - scaled_receiver.y + rm.y,
                z: (1 - 2 * k) as f64 * source.z - scaled_receiver.z + rm.z,
            };
            let refl = [
                beta.x[0].powi((mx - q).abs() as i32) * beta.x[1].powi((mx).abs() as i32),
                beta.y[0].powi((my - j).abs() as i32) * beta.y[1].powi((my).abs() as i32),
                beta.z[0].powi((mz - k).abs() as i32) * beta.z[1].powi((mz).abs() as i32),
            ];

            let dist = rp_plus_rm.radius();
            let fdist = dist.floor();

            if (fdist as usize) < n_samples
                && (n_order == -1
                    || (2 * mx - q).abs() + (2 * my - j).abs() + (2 * mz - k).abs() <= n_order)
            {
                let gain =
                    sim_microphone(rp_plus_rm, receiver) * refl[0] * refl[1] * refl[2]
                        / (4.0 * PI * dist * cts);

                let mut lpi = vec![0.0; tw];

                for (n, lpi) in lpi.iter_mut().enumerate() {
                    let t = ((n as f64) - 0.5 * (tw as f64) + 1.0) - (dist - fdist);
                    *lpi = 0.5
                        * (1.0 + (2.0 * PI * t / (tw as f64)).cos())
                        * 2.0
                        * FC
                        * (2.0 * PI * FC * t).sinc();
                }

                let start_position = (fdist - (tw as f64 / 2.0) + 1.0) as usize;

                for (imp, lpi) in imp[start_position..].iter_mut().zip(lpi) {
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

        for imp in imp.iter_mut() {
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
    use std::convert::identity;

    use crate::rir::*;

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

        assert!(imp
            .iter()
            .map(|inner| inner.iter().any(|&x| x > 0.0))
            .all(identity))
    }
}
