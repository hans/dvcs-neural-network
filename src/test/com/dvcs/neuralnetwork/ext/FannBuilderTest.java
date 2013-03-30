package com.dvcs.neuralnetwork.ext;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import com.dvcs.neuralnetwork.Example;
import com.dvcs.neuralnetwork.NeuralNetworkBuilder;
import com.dvcs.neuralnetwork.NeuralNetworkBuilder.DimensionMismatchException;
import com.dvcs.neuralnetwork.NeuralNetworkBuilder.InsufficientDataException;
import com.googlecode.fannj.Fann;

@RunWith(JUnit4.class)
public class FannBuilderTest {

	private FannBuilder builder;

	@Before
	public void setUp() {
		builder = new FannBuilder();
	}

	private void addExample(NeuralNetworkBuilder builder, float[] x, float[] y)
			throws DimensionMismatchException {
		builder.addExample(new Example(x, y));
	}

	@Test
	public void testSimpleBuild() throws DimensionMismatchException,
			InsufficientDataException {
		float[] input = new float[] { 3 };
		float[] output = new float[] { 1 };

		addExample(builder, input, output);
		Fann net = builder.buildFann(new int[] { 1 });

		float[] out = net.run(input);
		Assert.assertArrayEquals(output, out, 0.1f);
	}

	@Test(expected = InsufficientDataException.class)
	public void testInsufficientData() throws InsufficientDataException {
		builder.buildNetwork(new int[] { 1 }, 0);
	}

	@Test(expected = DimensionMismatchException.class)
	public void testDimensionMismatchException()
			throws DimensionMismatchException {
		addExample(builder, new float[] { 1, 2 }, new float[] { 1 });
		addExample(builder, new float[] { 3 }, new float[] { 1 });
	}

}
