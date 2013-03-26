package com.dvcs.neuralnetwork;

import java.util.ArrayList;
import java.util.List;

import org.jblas.DoubleMatrix;

/**
 * Training data for a neural network may not arrive all at once. This class
 * facilitates training data collection in real-time.
 * 
 * This class operates in as flexible a manner as possible with respect to
 * network topology. The sizes of the input and output layers will be inferred
 * from the first example provided if not explicitly specified in the
 * constructor. Hidden layer unit dimensions are specified only once a neural
 * network is about to be built.
 */
public class NeuralNetworkBuilder {

	private List<Example> examples;

	int inputLayerSize;
	int outputLayerSize;

	public NeuralNetworkBuilder() {
		// No network dimensions were specified. They will be inferred when the
		// first example is added.
		this(-1, -1);
	}

	public NeuralNetworkBuilder(int inputLayerSize, int outputLayerSize) {
		this.examples = new ArrayList<Example>();
		this.inputLayerSize = inputLayerSize;
		this.outputLayerSize = outputLayerSize;
	}

	public void addExample(Example example) throws DimensionMismatchException {
		// Do we have dimension data? If not, infer from this example. If so,
		// check for a match
		int inputSize = example.getX().length;
		if ( inputLayerSize == -1 )
			inputLayerSize = inputSize;
		else if ( inputLayerSize != inputSize )
			throw new DimensionMismatchException(
					"This example's input layer size doesn't match that of the network");

		int outputSize = example.getY().length;
		if ( outputLayerSize == -1 )
			outputLayerSize = outputSize;
		else if ( outputLayerSize != outputSize )
			throw new DimensionMismatchException(
					"This example's output layer size doesn't match that of the network");

		examples.add(example);
	}

	/**
	 * Build a neural network using the present data.
	 * 
	 * @param hiddenLayerSizes
	 *            The number of units in each hidden layer, proceeding in
	 *            natural order (i.e., the first element in the array represents
	 *            the size of the hidden layer immediately connected to the
	 *            input layer).
	 * @param lambda Regularization parameter
	 */
	public NeuralNetwork buildNetwork(int[] hiddenLayerSizes, double lambda)
			throws InsufficientDataException {
		if ( examples.size() == 0 )
			throw new InsufficientDataException();

		int[] layerSizes = new int[hiddenLayerSizes.length + 2];
		layerSizes[0] = inputLayerSize;
		layerSizes[layerSizes.length - 1] = outputLayerSize;
		System.arraycopy(hiddenLayerSizes, 0, layerSizes, 1,
				hiddenLayerSizes.length);

		NeuralNetwork ret = new NeuralNetwork(layerSizes);

		DoubleMatrix x = new DoubleMatrix(examples.size(), inputLayerSize);
		DoubleMatrix y = new DoubleMatrix(outputLayerSize, examples.size());
		
		for ( int i = 0; i < examples.size(); i++ ) {
			Example ex = examples.get(i);
			x.putRow(i, new DoubleMatrix(ex.getX()));
			y.putColumn(i, new DoubleMatrix(ex.getY()));
		}
		
		ret.train(x, y, lambda, null, false);
		return ret;
	}

	private class InsufficientDataException extends Exception {
	}

	private class DimensionMismatchException extends Exception {
		DimensionMismatchException(String message) {
			super(message);
		}
	}

	/**
	 * Represents a single example to be learned.
	 */
	public class Example {
		private double[] x;
		private double[] y;

		public Example(double[] x, double[] y) {
			this.x = x;
			this.y = y;
		}

		public double[] getX() {
			return x;
		}

		public double[] getY() {
			return y;
		}
	}

}
