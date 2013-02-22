package com.dvcs.tools;

import org.jblas.DoubleMatrix;

public class MatrixTools {

	public static DoubleMatrix copy(DoubleMatrix a) {
		return new DoubleMatrix(a.toArray2());
	}

	/**
	 * Reshape a vector into a square matrix. The vector must have a length that
	 * is a perfect square.
	 */
	public static DoubleMatrix reshape(double[] row) {
		double sq = Math.sqrt(row.length);
		if ((int) sq != sq) {
			throw new RuntimeException(
					"Square reshape method provided with a vector whose length is not a square number");
		}

		int length = (int) sq;
		return reshape(row, length, length);
	}

	/**
	 * Reshape a vector into a rectangular matrix.
	 * 
	 * @param row
	 *            Must have a length of `m * n`
	 * @param m
	 * @param n
	 */
	public static DoubleMatrix reshape(double[] row, int m, int n) {
		if (m * n != row.length) {
			throw new RuntimeException(
					"Reshape method provided with a vector whose length does not match the matrix size parameters");
		}

		DoubleMatrix ret = new DoubleMatrix(m, n);

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				ret.put(i, j, row[i * n + j]);
			}
		}

		return ret;
	}

	/**
	 * Replace each element x of a matrix with 1/(1+e^(-x)).
	 */
	public static DoubleMatrix matrixSigmoid(DoubleMatrix z) {
		z = copy(z);

		for (int i = 0; i < z.getRows(); i++) {
			for (int j = 0; j < z.getColumns(); j++) {
				z.put(i, j, sigmoid(z.get(i, j)));
			}
		}

		return z;
	}
	
	public static DoubleMatrix matrixSigmoidGradient(DoubleMatrix z) {
		z = copy(z);
		
		for ( int i = 0; i < z.getRows(); i++ ) {
			for ( int j = 0; j < z.getColumns(); j++ ) {
				double val = z.get(i, j);
				double sig = sigmoid(val);
				
				z.put(i, j, sig * (1 - sig));
			}
		}
		
		return z;
	}

	private static double sigmoid(double x) {
		return 1.0 / (1 + Math.exp(-x));
	}

	public static double matrixSum(DoubleMatrix m) {
		double sum = 0;

		for (int i = 0; i < m.getRows(); i++) {
			for (int j = 0; j < m.getColumns(); j++) {
				sum += m.get(i, j);
			}
		}

		return sum;
	}
}
