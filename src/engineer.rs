use std::ops::{AddAssign, Neg};

use num::{traits::Pow, Num};

use crate::polynomials::Polynomial;
pub fn create_polynomial_with_zeros<T: Num + Neg<Output = T> + PartialOrd + Copy+AddAssign>(
    zeros: Vec<T>,
    leading_coefficient: T,
) -> Polynomial<T> {
    // Create a neutral polynomial.
    let mut poly: Polynomial<T> = Polynomial::new_from_num_vec(vec![leading_coefficient]);
    for zero in zeros {
        let temp_poly = Polynomial::new_from_num_vec(vec![T::one(), -zero]);
        poly = poly * temp_poly
    }
    poly
}

pub fn create_quadratic_with_vertex<T: Num + Neg<Output = T> + Copy + PartialOrd+AddAssign>(
    vertex: (T, T),
    leading_coefficient: T,
) -> Polynomial<T> {
    (Polynomial::new_from_num_vec(vec![leading_coefficient])
        * Polynomial::new_from_num_vec(vec![T::one(), -vertex.0]).pow(2))
        + Polynomial::new_from_num_vec(vec![vertex.1])
}
