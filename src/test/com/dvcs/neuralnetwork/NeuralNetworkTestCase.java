package com.dvcs.neuralnetwork;

import org.jblas.DoubleMatrix;
import org.junit.Assert;
import org.junit.Test;

import com.dvcs.neuralnetwork.NeuralNetwork.ForwardPropagationResult;
import com.dvcs.tools.MatrixTools;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.dense.DenseDoubleVector;

public class NeuralNetworkTestCase {

	@Test
	public void testAddBiasUnit() {
		DoubleMatrix a = new DoubleMatrix(new double[][] {
				new double[] { 0.0, 1.0 }, new double[] { 2.0, 3.0 } });
		DoubleMatrix expected = new DoubleMatrix(new double[][] {
				new double[] { 1.0, 1.0 }, new double[] { 0.0, 1.0 },
				new double[] { 2.0, 3.0 } });

		Assert.assertEquals(expected, NeuralNetwork.addBiasUnit(a));
	}

	/**
	 * This "identity" network has the same number of units in each layer. Bias
	 * units are unweighted. The output should be second-order sigmoid of the
	 * input.
	 */
	@Test
	public void testIdentityWeightsFeedForward() {
		DoubleMatrix x = new DoubleMatrix(new double[][] { new double[] { 1.0,
				1.0, 1.0 } });

		DoubleMatrix Theta = new DoubleMatrix(new double[][] {
				new double[] { 0.0, 1.0, 0.0, 0.0 },
				new double[] { 0.0, 0.0, 1.0, 0.0 },
				new double[] { 0.0, 0.0, 0.0, 1.0 } });

		NeuralNetwork network = new NeuralNetwork(Theta, Theta);
		ForwardPropagationResult fResult = network.feedForward(x);

		DoubleMatrix expectedA2 = MatrixTools.matrixSigmoid(NeuralNetwork
				.addBiasUnit(x.transpose()));
		DoubleMatrix expectedA3 = MatrixTools.matrixSigmoid(expectedA2
				.getRange(1, expectedA2.getRows(), 0, expectedA2.getColumns()));

		Assert.assertEquals(expectedA2, fResult.getA2());
		Assert.assertEquals(expectedA3, fResult.getA3());
	}

	@Test
	public void testBuildYMatrix() {
		DoubleMatrix yVector = new DoubleMatrix(new double[][] {
				new double[] { 1.0 }, new double[] { 3.0 },
				new double[] { 2.0 } });

		DoubleMatrix expected = new DoubleMatrix(new double[][] {
				new double[] { 1.0, 0.0, 0.0 }, new double[] { 0.0, 0.0, 1.0 },
				new double[] { 0.0, 1.0, 0.0 } });

		Assert.assertEquals(expected, NeuralNetwork.buildYMatrix(yVector, 3));
	}

	@Test
	public void testConvertPointToWeightMatrices() {
		DoubleVector in = new DenseDoubleVector(new double[] { 1, 3, 2, 1, 3,
				2, 4, 8, 0, 5, 9, 10 });

		DoubleMatrix Theta1 = new DoubleMatrix(new double[][] {
				new double[] { 1, 2, 3 }, new double[] { 3, 1, 2 } });
		DoubleMatrix Theta2 = new DoubleMatrix(new double[][] {
				new double[] { 4, 5 }, new double[] { 8, 9 },
				new double[] { 0, 10 } });

		DoubleMatrix[] out = NeuralNetwork.convertPointToWeightMatrices(in,
				Theta1.getRows(), Theta1.getColumns(), Theta2.getRows(),
				Theta2.getColumns());
		Assert.assertArrayEquals(new DoubleMatrix[] { Theta1, Theta2 }, out);
	}

	@Test
	public void testConvertWeightMatricesToPoint() {
		DoubleMatrix Theta1 = new DoubleMatrix(new double[][] {
				new double[] { 1, 2, 3 }, new double[] { 3, 1, 2 } });
		DoubleMatrix Theta2 = new DoubleMatrix(new double[][] {
				new double[] { 4, 5 }, new double[] { 8, 9 },
				new double[] { 0, 10 } });

		DoubleVector expected = new DenseDoubleVector(new double[] { 1, 3, 2,
				1, 3, 2, 4, 8, 0, 5, 9, 10 });

		Assert.assertEquals(
				expected,
				NeuralNetwork.convertWeightMatricesToPoint(new DoubleMatrix[] {
						Theta1, Theta2 }));
	}
}
