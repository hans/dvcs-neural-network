package com.dvcs.neuralnetwork;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.junit.Assert;
import org.junit.Test;

import com.dvcs.neuralnetwork.NeuralNetwork;

public class NeuralNetworkTestCase {

	@Test
	public void testRandomMatrix() {
		int m = 5;
		int n = 10;
		
		RealMatrix mat = NeuralNetwork.randomMatrix(m, n, 0, 1);
		
		Assert.assertEquals(m, mat.getRowDimension());
		Assert.assertEquals(n, mat.getColumnDimension());
	}
	
	@Test
	public void testAddBiasUnit() {
		RealMatrix a = new Array2DRowRealMatrix(new double[][] {
				new double[] { 0.0, 1.0 }, new double[] { 2.0, 3.0 } });
		RealMatrix expected = new Array2DRowRealMatrix(new double[][] {
				new double[] { 1.0, 1.0 }, new double[] { 0.0, 1.0 },
				new double[] { 2.0, 3.0 } });

		Assert.assertEquals(expected, NeuralNetwork.addBiasUnit(a));
	}

	@Test
	public void testUnroll() {
		RealMatrix a = new Array2DRowRealMatrix(new double[][] {
				new double[] { 0.0, 1.0 }, new double[] { 2.0, 3.0 } });
		Double[] expected = new Double[] { 0.0, 1.0, 2.0, 3.0 };

		Assert.assertArrayEquals(expected,
				primitiveToBoxedDoubleArray(NeuralNetwork.unroll(a)));
	}

	static Double[] primitiveToBoxedDoubleArray(double[] xs) {
		Double[] ys = new Double[xs.length];

		for (int i = 0; i < xs.length; i++) {
			ys[i] = xs[i];
		}

		return ys;
	}

}
