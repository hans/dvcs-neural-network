package com.dvcs.tools;

import org.jblas.DoubleMatrix;
import org.junit.Assert;
import org.junit.Test;

public class MatrixToolsTestCase {

	@Test
	public void testReshapeSquare() {
		double[] in = new double[] { 1.0, 2.0, 3.0, 4.0 };
		DoubleMatrix expected = new DoubleMatrix(new double[][] {
				new double[] { 1.0, 3.0 }, new double[] { 2.0, 4.0 } });

		Assert.assertEquals(expected, MatrixTools.reshape(in));
	}

	@Test(expected = RuntimeException.class)
	public void testReshapeSquareException() {
		double[] in = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
		MatrixTools.reshape(in);
	}

	@Test
	public void testReshapeRectangular() {
		double[] in = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
		DoubleMatrix expected = new DoubleMatrix(new double[][] {
				new double[] { 1.0, 4.0 }, new double[] { 2.0, 5.0 },
				new double[] { 3.0, 6.0 } });

		Assert.assertEquals(expected, MatrixTools.reshape(in, 3, 2));
	}

	@Test(expected = RuntimeException.class)
	public void testReshapeRectangularException() {
		double[] in = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
		MatrixTools.reshape(in, 3, 2);
	}

}
