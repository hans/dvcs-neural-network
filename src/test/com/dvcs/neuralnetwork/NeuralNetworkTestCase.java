package com.dvcs.neuralnetwork;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.junit.Assert;
import org.junit.Test;

import com.dvcs.neuralnetwork.NeuralNetwork;
import com.dvcs.tools.MatrixTools;

public class NeuralNetworkTestCase {
	
	@Test
	public void testAddBiasUnit() {
		RealMatrix a = new Array2DRowRealMatrix(new double[][] {
				new double[] { 0.0, 1.0 }, new double[] { 2.0, 3.0 } });
		RealMatrix expected = new Array2DRowRealMatrix(new double[][] {
				new double[] { 1.0, 1.0 }, new double[] { 0.0, 1.0 },
				new double[] { 2.0, 3.0 } });

		Assert.assertEquals(expected, NeuralNetwork.addBiasUnit(a));
	}

	static Double[] primitiveToBoxedDoubleArray(double[] xs) {
		Double[] ys = new Double[xs.length];

		for (int i = 0; i < xs.length; i++) {
			ys[i] = xs[i];
		}

		return ys;
	}

}
