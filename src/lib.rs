#![feature(map_try_insert)]
//! A crate for working with polynomials easily.
/// A module for engineering polynomials.
///
/// It only has two functions because I forgot what I was going to do with it.
/// Please leave suggestions in the Github issues.
pub mod engineer;
/// An implementation of polynomial arithmetic with all four basic operations and the `pow` trait.
///
/// # Example usage
/// ## Create a new polynomial from a vec of numbers.
/// ```
/// use polynomials_pro::polynomials::*;
/// let poly=Polynomial::new_from_num_vec(vec![4,5,6,3]);
/// println!("{}",poly); // 4x^3+5x^2+6x+3
/// ```
/// ## Multiply polynomials
/// ```
/// use polynomials_pro::polynomials::*;
/// let poly1 = Polynomial::new_from_num_vec(vec![1, 5, 6]);
/// let poly2 = Polynomial::new_from_num_vec(vec![5, 6, 9, 6]);
/// assert_eq!(poly1 * poly2,Polynomial::new_from_num_vec(vec![5,31,69,87,84,36]));
/// ```
pub mod polynomials {
    use num::{Float, Num, Signed, Zero};
    use std::{
        collections::{BTreeMap, HashSet}, fmt::Display, ops::{Add, AddAssign, Div, DivAssign, Mul, Sub}
    };
    #[derive(Debug, Clone, PartialEq)]
    pub struct Polynomial<T: Num>(pub(crate) Vec<PolynomialTerm<T>>);
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct PolynomialTerm<T: Num> {
        pub power: u32,
        pub coefficient: T,
    }
    impl<T: Num + PartialOrd + AddAssign + Copy> Polynomial<T> {
        pub(crate) fn combine_like_terms(&mut self) {
            let mut term_map = BTreeMap::new();
            for term in self.0.iter() {
                if term.coefficient.is_zero() {
                    continue;
                }
                if let Err(occupied) = term_map.try_insert(term.power, term.coefficient) {
                    *occupied.entry.into_mut() = occupied.value + *occupied.entry.get();
                }
            }
            self.0.clear();
            for pair in term_map {
                self.0.push(PolynomialTerm {
                    power: pair.0,
                    coefficient: pair.1,
                });
            }
            self.0.sort_by(|a, b| b.power.cmp(&a.power));
        }
    }
    impl<T: Num + PartialOrd + AddAssign> Polynomial<T> {
        // potential zero-copy like terms algorithm
        // i have debugged this for a long time and tried to find differences between outputs from this function and outputs from the current combine like terms fn
        // but i cannot find any
        // for some reason, using this puts polynomial-by-polynomial division in an infinite loop
        pub(crate) fn _new_combine_like_terms(&mut self) {
            loop {
                let mut seen = HashSet::new();
                let mut found_duplicates = false;
                for term in &self.0 {
                    if !seen.insert(term.power) {
                        found_duplicates = true;
                        break;
                    }
                }
                if !found_duplicates {
                    break;
                }
                let adding_term = match self.0.pop() {
                    Some(term) => term,
                    None => return,
                };
                if adding_term.coefficient.is_zero() {
                    continue;
                }
                let our_len = self.0.len();
                for (i, term) in self.0.iter_mut().enumerate() {
                    if term.power == adding_term.power {
                        term.coefficient += adding_term.coefficient;
                        break;
                    } else if i == our_len - 1 {
                        self.0.insert(0, adding_term);
                        break;
                    }
                }
            }
            self.0.sort_by(|a, b| b.power.cmp(&a.power));
        }
    }

    impl<T: Num + PartialOrd + AddAssign + Copy> Polynomial<T> {
        /// Pushes a term to the polynomial and sorts the terms into PartialOrder.
        pub fn push_term(&mut self, term: PolynomialTerm<T>) {
            self.0.push(term);
            self.combine_like_terms();
        }
    }
    impl<T: Num> Polynomial<T> {
        pub fn get_degree(&self) -> u32 {
            let mut max = 0;
            for term in &self.0 {
                if term.coefficient.is_zero() {
                    continue;
                }
                if term.power > max {
                    max = term.power
                }
            }
            max
        }
    }
    impl<T: Num + PartialOrd + AddAssign + Copy> Polynomial<T> {
        pub fn new_from_term_vec(terms: Vec<PolynomialTerm<T>>) -> Self {
            let mut new = Polynomial(Vec::new());
            for term in terms {
                new.push_term(term);
            }
            new
        }
    }
    impl<T: Num + Copy + PartialOrd + AddAssign> Polynomial<T> {
        /// This function takes a vector of numbers, where each element has a power one less than the element before it, with the first element having the power of the length of the vector.
        pub fn new_from_num_vec(terms: Vec<T>) -> Self {
            if terms.is_empty() {
                return Self(vec![]);
            }
            let degree = terms.len() - 1;
            let mut polynomial_terms: Vec<PolynomialTerm<T>> = Vec::new();
            for (i, term) in terms.iter().enumerate() {
                polynomial_terms.push(PolynomialTerm {
                    power: (degree - i) as u32,
                    coefficient: *term,
                });
            }
            let mut poly = Self(polynomial_terms);
            poly.combine_like_terms();
            poly
        }
    }
    impl<T: Num + Copy + PartialOrd + Signed + DivAssign + Float + AddAssign> Polynomial<T> {
        pub fn new_from_points(points: Vec<(T, T)>) -> Self {
            let mut poly = Polynomial::new_from_num_vec(vec![]);
            for point in &points {
                let mut delta = Polynomial::new_from_num_vec(vec![point.1]);
                for other_point in &points {
                    if other_point.0 == point.0 {
                        continue;
                    }
                    delta = delta
                        * (Polynomial::new_from_term_vec(vec![
                            PolynomialTerm {
                                coefficient: T::one(),
                                power: 1,
                            },
                            PolynomialTerm {
                                coefficient: -other_point.0,
                                power: 0,
                            },
                        ]) / (point.0 - other_point.0));
                }
                poly = poly + delta.clone();
            }
            poly
        }
    }
    impl<T: Num + Copy + PartialOrd + num::pow::Pow<u32, Output = T> + AddAssign> Polynomial<T> {
        pub fn evaluate(&mut self, x: T) -> T {
            let mut answer = T::zero();
            for term in &self.0 {
                answer += term.coefficient * x.pow(term.power);
            }
            answer
        }
    }
    impl<T: Num + Copy + PartialOrd + AddAssign> Zero for Polynomial<T> {
        fn zero() -> Self {
            Polynomial::new_from_num_vec(vec![])
        }

        fn is_zero(&self) -> bool {
            let mut total = T::zero();
            for term in &self.0 {
                total += term.coefficient;
            }
            total.is_zero()
        }
    }
    impl<T: Num + Copy + PartialOrd + AddAssign> num::One for Polynomial<T> {
        fn one() -> Self {
            Polynomial::new_from_num_vec(vec![T::one()])
        }
    }

    impl<T: Num + Signed + Display> Display for Polynomial<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let mut formatted_equation = String::new();
            let mut actual_i = 0;
            for term in self.0.iter() {
                if term.coefficient.is_zero() {
                    continue;
                }
                if actual_i != 0 && term.coefficient.is_positive() {
                    formatted_equation += "+"
                }
                if !term.coefficient.is_one() || term.power == 0 {
                    formatted_equation += &term.coefficient.to_string();
                }
                if term.power != 0 {
                    formatted_equation += "x";
                    if term.power != 1 {
                        formatted_equation += &format!("^{}", term.power);
                    }
                }
                actual_i += 1;
            }
            write!(f, "{}", formatted_equation)
        }
    }
    impl<T: Num + PartialOrd + AddAssign + Copy> Mul<Polynomial<T>> for Polynomial<T> {
        type Output = Polynomial<T>;

        fn mul(self, rhs: Polynomial<T>) -> Self::Output {
            let mut out = Polynomial::new_from_term_vec(Vec::new());

            for lht in self.0 {
                for rht in rhs.0.iter() {
                    out.push_term(lht * (*rht));
                }
            }
            out.combine_like_terms();
            out
        }
    }

    impl<T: Num> Mul<PolynomialTerm<T>> for PolynomialTerm<T> {
        type Output = PolynomialTerm<T>;

        fn mul(self, rhs: PolynomialTerm<T>) -> Self::Output {
            PolynomialTerm {
                power: self.power + rhs.power,
                coefficient: self.coefficient * rhs.coefficient,
            }
        }
    }
    impl<T: Num + Copy> Mul<PolynomialTerm<T>> for Polynomial<T> {
        type Output = Polynomial<T>;

        fn mul(self, rhs: PolynomialTerm<T>) -> Self::Output {
            let out = self.clone().0.iter().map(|x| *x * rhs).collect();
            Polynomial(out)
        }
    }

    impl<T: Num + Copy> Mul<T> for Polynomial<T> {
        type Output = Polynomial<T>;

        fn mul(self, rhs: T) -> Self::Output {
            self * PolynomialTerm {
                coefficient: rhs,
                power: 0,
            }
        }
    }
    impl<T: Num + Copy + PartialOrd + AddAssign> Add for Polynomial<T> {
        type Output = Polynomial<T>;

        fn add(self, rhs: Self) -> Self::Output {
            let mut cloned = self.clone();
            cloned.0.extend(rhs.0);
            cloned.combine_like_terms();
            cloned
        }
    }

    impl<T: Num + Copy + Signed + PartialOrd + AddAssign> Sub for Polynomial<T> {
        type Output = Polynomial<T>;

        fn sub(self, rhs: Self) -> Self::Output {
            // Negate all terms in rhs then add to polynomial
            let mut cloned = rhs.clone();
            for term in cloned.0.iter_mut() {
                term.coefficient = -term.coefficient
            }
            self + cloned
        }
    }
    pub struct DivOutput<T: Num> {
        pub remainder: Polynomial<T>,
        pub output: Polynomial<T>,
    }
    impl<T: Num> Div for PolynomialTerm<T> {
        type Output = PolynomialTerm<T>;

        fn div(self, rhs: Self) -> Self::Output {
            Self {
                power: self.power - rhs.power,
                coefficient: self.coefficient / rhs.coefficient,
            }
        }
    }
    impl<T: Num + Copy> Div<PolynomialTerm<T>> for Polynomial<T> {
        type Output = Polynomial<T>;

        fn div(self, rhs: PolynomialTerm<T>) -> Self::Output {
            let mut out = self;
            for term in out.0.iter_mut() {
                *term = *term / rhs;
            }
            out
        }
    }
    impl<T: Num + Copy + PartialOrd + AddAssign> num::pow::Pow<i32> for Polynomial<T> {
        type Output = Polynomial<T>;

        fn pow(self, rhs: i32) -> Self::Output {
            let mut poly = Polynomial::new_from_num_vec(vec![T::one()]);
            for _ in 0..rhs {
                poly = poly * self.clone();
            }
            todo!()
        }
    }
    impl<T: Num + Display + Copy + PartialOrd + Signed + AddAssign> Display for PolynomialTerm<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let poly = Polynomial::new_from_term_vec(vec![*self]);

            write!(f, "{}", poly)
        }
    }
    impl<T: Num + Copy + PartialOrd + Signed + AddAssign> Div<Polynomial<T>>
        for Polynomial<T>
    {
        type Output = DivOutput<T>;
        // Implementation of polynomial long division.
        fn div(self, rhs: Self) -> Self::Output {
            let first_divisor_term = rhs.0.first().expect("cannot divide by empty polynomial");
            let mut answer = Polynomial::new_from_term_vec(Vec::new());
            let mut remainder = self;
            while remainder.get_degree() >= rhs.get_degree() {
                let factor = *remainder.0.first().unwrap() / *first_divisor_term;
                answer.push_term(factor);
                remainder = remainder.clone() - (rhs.clone() * factor);
            }

            remainder.combine_like_terms();
            answer.combine_like_terms();
            DivOutput {
                remainder,
                output: answer,
            }
        }
    }
    impl<T: Num + Copy + DivAssign> Div<T> for Polynomial<T> {
        type Output = Polynomial<T>;

        fn div(mut self, rhs: T) -> Self::Output {
            if rhs.is_zero() {
                panic!("cannot divide polynomial by zero");
            }
            for term in self.0.iter_mut() {
                term.coefficient /= rhs
            }
            self
        }
    }
}
#[cfg(test)]
mod tests {
    //use rand::Rng;

    use super::polynomials::*;
    #[test]
    fn polynomial_div() {
        let poly1 = Polynomial::new_from_num_vec(vec![1., 5., 6.]);
        let poly2 = Polynomial::new_from_num_vec(vec![5., 6., 6.]);
        let combined = poly1.clone() * poly2.clone();
        let divided = combined / poly1;
        assert_eq!(divided.output, poly2);
    }
    #[test]
    fn polynomial_div_with_remainder() {
        let dividend = Polynomial::new_from_num_vec(vec![1., -12., 38., -17.]);
        let divisor = Polynomial::new_from_num_vec(vec![3., 18., 14.]);
        assert_eq!(
            (dividend / divisor).remainder,
            Polynomial::new_from_num_vec(vec![424. / 3., 67.])
        )
    }
    #[test]
    fn polynomial_mul() {
        let poly1 = Polynomial::new_from_num_vec(vec![1, 5, 6]);
        let poly2 = Polynomial::new_from_num_vec(vec![5, 6, 9, 6]);
        assert_eq!(
            poly1 * poly2,
            Polynomial::new_from_num_vec(vec![5, 31, 69, 87, 84, 36])
        );
    }
    #[test]
    fn poly_from_points() {
        let poly: Polynomial<f32> = Polynomial::new_from_num_vec(vec![5., 6., 9., 6.]);
        let points: Vec<(f32, f32)> =
            vec![(9.0, 4218.0), (3.0, 222.0), (4.0, 458.0), (7.0, 2078.0)];
        let new_poly = Polynomial::new_from_points(points);
        assert_eq!(new_poly, poly)
    }

    #[test]
    fn create_new(){
        let poly=Polynomial::new_from_num_vec(vec![-4., 0., 64.]);
        println!("{}",poly);
    }
    /*
    #[test]
    fn new_vs_old(){
        loop{
            let mut poly=random_terms();
            let mut poly2=random_terms();
            poly.0.append(&mut poly2.0);
            let mut old=poly.clone();
            assert_eq!(poly.combine_like_terms(),old.new_combine_like_terms());
        }
    }
    fn random_terms()->Polynomial<i32>{
        let rng = &mut rand::thread_rng();
        let term_amount=rng.gen_range(1..3);
        let mut terms=vec![];
        for _ in 0..term_amount{
            let coef=rng.gen_range(-10..=10);
            if coef==0{
                continue
            }
            terms.push(PolynomialTerm{
                coefficient: coef,
                power: rng.gen_range(1..5)
            })
        }
        Polynomial(terms)

    }*/
}
