package com.dvcs.tools;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;


public class MatrixTools {

	/**
	 * Replace each element x of a matrix with 1/(1+e^(-x)).
	 */
	public static RealMatrix matrixSigmoid(RealMatrix z) {
		z = z.copy();
	
		for (int i = 0; i < z.getRowDimension(); i++) {
			for (int j = 0; j < z.getColumnDimension(); j++) {
				z.setEntry(i, j, MatrixTools.sigmoid(z.getEntry(i, j)));
			}
		}
	
		return z;
	}

	private static double sigmoid(double x) {
		return 1.0 / (1 + Math.exp(-x));
	}

	public static double matrixSum(RealMatrix m) {
		double sum = 0;
	
		for (int i = 0; i < m.getRowDimension(); i++) {
			for (int j = 0; j < m.getColumnDimension(); j++) {
				sum += m.getEntry(i, j);
			}
		}
	
		return sum;
	}

	/**
	 * Square each element of a matrix.
	 */
	public static RealMatrix squareMatrixElements(RealMatrix m) {
		m = m.copy();
	
		for (int i = 0; i < m.getRowDimension(); i++) {
			for (int j = 0; j < m.getColumnDimension(); j++) {
				double val = m.getEntry(i, j);
				m.setEntry(i, j, val * val);
			}
		}
	
		return m;
	}

	/**
	 * Replace each element $x$ of a matrix with $\log(x)$.
	 */
	static RealMatrix matrixLogarithm(RealMatrix z) {
		z = z.copy();
	
		for (int i = 0; i < z.getRowDimension(); i++) {
			for (int j = 0; j < z.getColumnDimension(); j++) {
				z.setEntry(i, j, Math.log(z.getEntry(i, j)));
			}
		}
	
		return z;
	}

	/**
	 * Build an `m` by `n` matrix where each cell's value is a random decimal
	 * number between `min` (inclusive) and `max` (exclusive).
	 */
	public static RealMatrix randomMatrix(int m, int n, double min, double max) {
		RealMatrix ret = new Array2DRowRealMatrix(m, n);
	
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				double val = Math.random() * (max - min) + min;
				ret.setEntry(i, j, val);
			}
		}
	
		return ret;
	}

	/**
	 * "Unroll" a matrix into a single-dimensional array.
	 */
	public static double[] unroll(RealMatrix m) {
		double[] ret = new double[m.getRowDimension() * m.getColumnDimension()];
		
		for ( int i = 0; i < m.getRowDimension(); i++ ) {
			for ( int j = 0; j < m.getColumnDimension(); j++ ) {
				ret[i * m.getRowDimension() + j] = m.getEntry(i, j);
			}
		}
		
		return ret;
	}

	/**
	 * Unroll a vector into a square matrix. The vector must have a length that
	 * is a perfect square.
	 */
	public static RealMatrix reshape(double[] row) {
		double sq = Math.sqrt(row.length);
		if ((int) sq != sq) {
			throw new RuntimeException(
					"Reshape method provided with a vector whose length is not a square number");
		}
		
		int length = (int) sq;
		RealMatrix ret = new Array2DRowRealMatrix(length, length);
		
		for ( int i = 0; i < length; i++ ) {
			for ( int j = 0; j < length; j++ ) {
				ret.setEntry(i, j, row[i * length + j]);
			}
		}
		
		return ret;
	}

}
