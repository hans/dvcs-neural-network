package com.dvcs.neuralnetwork;

import org.jblas.DoubleMatrix;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import com.dvcs.neuralnetwork.NeuralNetwork.ForwardPropagationResult;
import com.dvcs.neuralnetwork.NeuralNetworkBuilder.DimensionMismatchException;
import com.dvcs.neuralnetwork.NeuralNetworkBuilder.InsufficientDataException;

@RunWith(JUnit4.class)
public class NeuralNetworkBuilderTest {

	private NeuralNetworkBuilder builder;

	/**
	 * A builder created with preset dimensions
	 */
	private NeuralNetworkBuilder builder2;

	@Before
	public void setUp() {
		builder = new NeuralNetworkBuilder();
		builder2 = new NeuralNetworkBuilder(2, 2);
	}

	private void addExample(NeuralNetworkBuilder builder, double[] x, double[] y)
			throws DimensionMismatchException {
		builder.addExample(new Example(x, y));
	}

	@Test
	public void testSimpleBuild() throws DimensionMismatchException,
			InsufficientDataException {
		double[] input = new double[] { 3 };
		double[] output = new double[] { 1 };

		DoubleMatrix inputM = new DoubleMatrix(1, input.length, input);

		addExample(builder, input, output);
		NeuralNetwork net = builder.buildNetwork(new int[] { 1 }, 0);

		ForwardPropagationResult fResult = net.feedForward(inputM);
		Assert.assertArrayEquals(output, fResult.getOutputLayer().toArray(),
				0.001);
	}

	@Test(expected = InsufficientDataException.class)
	public void testInsufficientData() throws InsufficientDataException {
		builder.buildNetwork(new int[] { 1 }, 0);
	}

	@Test(expected = DimensionMismatchException.class)
	public void testDimensionMismatchException()
			throws DimensionMismatchException {
		addExample(builder, new double[] { 1, 2 }, new double[] { 1 });
		addExample(builder, new double[] { 3 }, new double[] { 1 });
	}

}
