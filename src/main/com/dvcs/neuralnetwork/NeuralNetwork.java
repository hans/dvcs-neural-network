package com.dvcs.neuralnetwork;

import java.util.Arrays;

import org.jblas.DoubleMatrix;

import com.dvcs.tools.MatrixTools;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.dense.DenseDoubleVector;
import de.jungblut.math.minimize.Fmincg;
import de.jungblut.math.minimize.MinimizerListener;

public class NeuralNetwork {

	final double RANDOM_WEIGHT_MATRIX_MINIMUM = 0;
	final double RANDOM_WEIGHT_MATRIX_MAXIMUM = 0.12;

	DoubleMatrix[] Thetas;

	/**
	 * For an element $i$, the number of rows in the matrix $\Theta^{(i)}$.
	 */
	int[] rowDimensions;

	/**
	 * For an element $i$, the number of columns in the matrix $\Theta^{(i)}$.
	 */
	int[] columnDimensions;

	/**
	 * @param layerSizes
	 *            An array representing the sequence of layers in the network.
	 *            The value of each element represents the number of units in
	 *            the corresponding layer.
	 */
	public NeuralNetwork(int[] layerSizes) {
		Thetas = new DoubleMatrix[layerSizes.length - 1];
		rowDimensions = new int[layerSizes.length - 1];
		columnDimensions = new int[layerSizes.length - 1];

		for (int i = 0; i < layerSizes.length - 1; i++) {
			int rows = layerSizes[i + 1];
			int columns = layerSizes[i] + 1;

			Thetas[i] = DoubleMatrix
					.rand(rows, columns)
					.mul(RANDOM_WEIGHT_MATRIX_MAXIMUM
							- RANDOM_WEIGHT_MATRIX_MINIMUM)
					.add(RANDOM_WEIGHT_MATRIX_MINIMUM);

			rowDimensions[i] = rows;
			columnDimensions[i] = columns;
		}
	}

	public NeuralNetwork(DoubleMatrix[] Thetas) {
		this.Thetas = Thetas;

		rowDimensions = new int[Thetas.length];
		columnDimensions = new int[Thetas.length];

		for (int i = 0; i < Thetas.length; i++) {
			rowDimensions[i] = Thetas[i].getRows();
			columnDimensions[i] = Thetas[i].getColumns();
		}
	}

	/**
	 * Train using a vector of output unit indices `y` rather than a matrix of
	 * output unit values (as in the normal `train`).
	 * 
	 * This method should only be used (and really can only be used) when the
	 * actual choices we want the network to make are *binary*: i.e., that we
	 * ideally only want one output value to be "confident" for any given
	 * forward propagation.
	 * 
	 * The `y` vector should be a list of the single units chosen for each
	 * example.
	 */
	public void train(DoubleMatrix x, DoubleMatrix yVector, int k,
			double lambda, MinimizerListener listener,
			boolean updateThetasDuringOptimization) {
		DoubleMatrix Y = buildYMatrix(yVector, k);
		train(x, Y, lambda, listener, updateThetasDuringOptimization);
	}

	/**
	 * Rebuild the weights of this network to minimize the error on the given
	 * data set.
	 * 
	 * In a network with desired binary output values, the output matrix will
	 * consist of columns from an identity matrix.
	 * 
	 * @param x
	 *            Matrix of examples (where each row represents an example and
	 *            each column represents a unit)
	 * @param y
	 *            Optimal output unit values (where each row is a unit and each
	 *            column is an example)
	 * @param lambda
	 *            Regularization parameter
	 * @param listener
	 *            A listener which will receive information about each
	 *            minimization iteration
	 * @param updateThetasDuringOptimization
	 *            If true, this network's parameters will update in sync with
	 *            the optimization steps. (The network can predict in real time
	 *            as it self-optimizes.) If false, the network's parameters
	 *            won't be updated until optimization has finished.
	 */
	public void train(DoubleMatrix x, DoubleMatrix y, double lambda,
			final MinimizerListener listener,
			boolean updateThetasDuringOptimization) {
		if (x.getRows() != y.getColumns()) {
			throw new RuntimeException(
					"Output matrix ss do not correspond with those of the example matrix");
		}

		ForwardPropagationResult fResult = feedForward(x);
		DoubleMatrix outputLayer = fResult.getOutputLayer();

		if (outputLayer.getRows() != y.getRows()
				|| outputLayer.getColumns() != y.getColumns()) {
			throw new RuntimeException(
					"Given output matrix ss do not correspond with those of the actual output matrix produced by the network");
		}

		MinimizerListener ourListener = null;
		if (updateThetasDuringOptimization) {
			ourListener = new MinimizerListener() {
				public void minimizationIterationFinished(int n, double cost,
						DoubleVector parameters) {
					Thetas = convertPointToWeightMatrices(parameters);

					if (listener != null) {
						listener.minimizationIterationFinished(n, cost,
								parameters);
					}
				}
			};
		} else {
			ourListener = listener;
		}

		// Unroll Theta1 and Theta2, then concatenate. This forms our initial
		// parameter vector.
		DoubleVector initParams = convertWeightMatricesToPoint(Thetas);

		NeuralNetworkCostFunction cost = new NeuralNetworkCostFunction(this, x,
				y, lambda);
		DoubleVector parameters = Fmincg.minimizeFunction(cost, initParams,
				100, ourListener);

		Thetas = convertPointToWeightMatrices(parameters);
	}

	public DoubleMatrix[] backpropagate(ForwardPropagationResult fResult,
			DoubleMatrix y) {
		return backpropagate(fResult, y, Thetas);
	}

	/**
	 * Determine what changes should be made to the weight matrices to lower
	 * cost on the given data set.
	 */
	public DoubleMatrix[] backpropagate(ForwardPropagationResult fResult,
			DoubleMatrix y, DoubleMatrix[] Thetas) {
		int m = y.getColumns();

		// If we have $l$ layers in the network, this array should have $l -
		// 1$ elements. The first element corresponds to `Delta1`, the total
		// change necessary in `Theta1`.
		DoubleMatrix[] Deltas = new DoubleMatrix[Thetas.length];

		DoubleMatrix[] preLayerValues = fResult.getPreLayerValues();
		DoubleMatrix[] layerValues = fResult.getLayerValues();

		for (int i = 0; i < m; i++) {
			DoubleMatrix[] deltaVectors = new DoubleMatrix[Thetas.length];

			// The first "error" values are actual residuals.
			deltaVectors[deltaVectors.length - 1] = fResult.getOutputLayer()
					.getColumn(i).sub(y.getColumn(i));

			// Propagate through the hidden layers, using error values and
			// parameters (Theta) to build a weighted sum. We begin with $l$
			// equivalent to the index of the last hidden layer in the network,
			// and $l$ moves backward to the first hidden layer $l = 1$.
			for (int l = layerValues.length - 2; l > 0; l--) {
				DoubleMatrix deltaL = Thetas[l].transpose().mmul(
						deltaVectors[l]);

				// Remove the error corresponding to the bias unit.
				deltaL = deltaL.getRange(1, deltaL.getRows(), 0,
						deltaL.getColumns());

				deltaL = deltaL.mul(MatrixTools
						.matrixSigmoidGradient(preLayerValues[l - 1]
								.getColumn(i)));

				deltaVectors[l - 1] = deltaL;
			}

			// We've collected all of our delta vectors; now build the final
			// weight shifting matrices.
			for (int l = 0; l < Thetas.length; l++) {
				DoubleMatrix Delta = deltaVectors[l].mmul(layerValues[l]
						.getColumn(i).transpose());
				
				if ( Deltas[l] == null ) {
					Deltas[l] = Delta;
				} else {
					Deltas[l] = Deltas[l].add(Delta);
				}
			}
		}

		return Deltas;
	}

	public class ForwardPropagationResult {
		/**
		 * The preliminary value of each layer before a bias layer has been
		 * added and before the activation function has been applied. The first
		 * element of this matrix corresponds to the *second* layer's pre-layer
		 * values.
		 */
		private DoubleMatrix[] preLayerValues;

		/**
		 * The values of each unit for each example in each layer.
		 */
		private DoubleMatrix[] layerValues;

		public ForwardPropagationResult(DoubleMatrix[] preLayerValues,
				DoubleMatrix[] layerValues) {
			this.preLayerValues = preLayerValues;
			this.layerValues = layerValues;
		}

		public DoubleMatrix[] getPreLayerValues() {
			return preLayerValues;
		}

		public DoubleMatrix[] getLayerValues() {
			return layerValues;
		}

		public DoubleMatrix getOutputLayer() {
			DoubleMatrix[] layers = getLayerValues();
			return layers[layers.length - 1];
		}

		/**
		 * @param layer
		 *            Zero-based layer index
		 * @return Matrix of unit values for this layer
		 */
		public DoubleMatrix getSingleLayerValues(int layer) {
			return getLayerValues()[layer];
		}

		/**
		 * @param layer
		 *            Zero-based layer index
		 * @return Matrix of pre-activation + pre-bias values for this layer
		 */
		public DoubleMatrix getSinglePreLayerValues(int layer) {
			if (layer == 0) {
				throw new RuntimeException(
						"Pre-layer values do not exist for the input layer.");
			}

			return getPreLayerValues()[layer - 1];
		}

		public DoubleMatrix[] getHiddenLayerValues() {
			DoubleMatrix[] layerValues = getLayerValues();
			DoubleMatrix[] ret = new DoubleMatrix[layerValues.length - 2];

			for (int l = 1; l < layerValues.length - 1; l++) {
				ret[l - 1] = layerValues[l];
			}

			return ret;
		}
	}

	public ForwardPropagationResult feedForward(DoubleMatrix x) {
		return feedForward(x, Thetas);
	}

	/**
	 * Feed a collection of examples forward through the network.
	 * 
	 * @param x
	 *            Example matrix, where each row represents an example and each
	 *            column represents a unit
	 * @param Theta1
	 *            A weight matrix describing the transformation of the input
	 *            data to hidden layer data
	 * @param Theta2
	 *            A weight matrix describing the transformation of the hidden
	 *            layer data to the output layer data
	 * @return A matrix of output layer units, where each row represents a unit
	 *         and each column represents an example
	 */
	public ForwardPropagationResult feedForward(DoubleMatrix x,
			DoubleMatrix[] Thetas) {
		DoubleMatrix[] preLayerValues = new DoubleMatrix[Thetas.length];
		DoubleMatrix[] layerValues = new DoubleMatrix[Thetas.length + 1];

		layerValues[0] = addBiasUnit(x.transpose());

		for (int l = 0; l < Thetas.length; l++) {
			preLayerValues[l] = Thetas[l].mmul(layerValues[l]);

			DoubleMatrix withBias = l == Thetas.length - 1 ? preLayerValues[l]
					: addBiasUnit(preLayerValues[l]);

			layerValues[l + 1] = MatrixTools.matrixSigmoid(withBias);
		}

		return new ForwardPropagationResult(preLayerValues, layerValues);
	}

	/**
	 * Given a matrix of examples (where each row represents a unit and each
	 * column represents an example), return an array where each element
	 * corresponds to an example and its value is the index of the class whose
	 * corresponding output unit has the highest value.
	 */
	public int[] predict(DoubleMatrix a3) {
		int[] results = new int[a3.getColumns()];

		for (int j = 0; j < a3.getColumns(); j++) {
			results[j] = maxIndex(a3.getColumn(j).toArray());
		}

		return results;
	}

	/**
	 * Return the index of the element in the array with the maximum value.
	 */
	public static int maxIndex(double[] xs) {
		int maxIndex = -1;
		double max = Double.MIN_VALUE;

		for (int i = 0; i < xs.length; i++) {
			double x = xs[i];

			if (x > max) {
				max = x;
				maxIndex = i;
			}
		}

		return maxIndex;
	}
	
	/**
	 * Return the index of the element in the array with the maximum value.
	 */
	public static int maxIndex(float[] xs) {
		int maxIndex = -1;
		float max = Float.MIN_VALUE;

		for (int i = 0; i < xs.length; i++) {
			float x = xs[i];

			if (x > max) {
				max = x;
				maxIndex = i;
			}
		}

		return maxIndex;
	}

	/**
	 * Add a bias unit to a layer (represented by a matrix). In this layer each
	 * column represents an individual example.
	 */
	static DoubleMatrix addBiasUnit(DoubleMatrix a) {
		DoubleMatrix ret = new DoubleMatrix(a.getRows() + 1, a.getColumns());

		// Initialize the bias unit (first row)
		for (int j = 0; j < ret.getColumns(); j++) {
			ret.put(0, j, 1);
		}

		// Place the original layer data below this first row
		for (int i = 0; i < a.getRows(); i++) {
			ret.putRow(i + 1, a.getRow(i));
		}

		return ret;
	}

	/**
	 * The output class answers are saved as a $y$ vector of length m (where
	 * cell $i$ has a certain output unit index for example $i$). The neural
	 * network implementation requires a full $Y$ matrix, where each example has
	 * its own column (where each row is the value for a given unit).
	 * 
	 * Each column should look like the column of an identity matrix. This
	 * function converts from the vector form to the matrix form.
	 * 
	 * @param yVector
	 * @param k
	 *            The number of output units in the network. (This determines
	 *            the number of rows in the Y matrix.)
	 * 
	 *            Every value in `yVector` should be between `1` and `k`
	 *            (inclusive).
	 */
	static DoubleMatrix buildYMatrix(DoubleMatrix yVector, int k) {
		DoubleMatrix ret = new DoubleMatrix(k, yVector.getRows());

		for (int i = 0; i < yVector.getRows(); i++) {
			int exampleOutput = (int) yVector.get(i, 0);

			if (exampleOutput > k) {
				throw new RuntimeException(
						"Example in output vector has an index greater than `k`");
			}

			ret.put(exampleOutput - 1, i, 1);
		}

		return ret;
	}

	DoubleMatrix[] convertPointToWeightMatrices(DoubleVector point) {
		return convertPointToWeightMatrices(point, rowDimensions,
				columnDimensions);
	}

	/**
	 * A "point" provided by the optimization algorithm consists of a series of
	 * structures, each of which contains a single parameter value. We can turn
	 * this into a flat list of parameters.
	 * 
	 * Reshape the flat parameter list required by the optimization method into
	 * the multiple separate matrices that it represents (i.e., Theta1, Theta2,
	 * and so on unrolled and then concatenated).
	 */
	static DoubleMatrix[] convertPointToWeightMatrices(DoubleVector point,
			int[] rowDimensions, int[] columnDimensions) {
		assert rowDimensions.length == columnDimensions.length;

		double[] params = point.toArray();

		DoubleMatrix[] ret = new DoubleMatrix[rowDimensions.length];
		int cursor = 0;

		for (int l = 0; l < ret.length; l++) {
			int ThetaLength = rowDimensions[l] * columnDimensions[l];
			double[] ThetaUnrolled = Arrays.copyOfRange(params, cursor, cursor
					+ ThetaLength);
			ret[l] = MatrixTools.reshape(ThetaUnrolled, rowDimensions[l],
					columnDimensions[l]);

			cursor += ThetaLength;
		}

		return ret;
	}

	/**
	 * Convert weight matrices into a single "point" for use with the
	 * optimization algorithm.
	 * 
	 * Unroll and concatenate the two weight matrices into one long vector.
	 */
	static DoubleVector convertWeightMatricesToPoint(DoubleMatrix[] weights) {
		DoubleMatrix Theta1 = weights[0];
		DoubleMatrix Theta2 = weights[1];

		// Unroll the matrices.
		double[] Theta1Unrolled = Theta1.toArray();
		double[] Theta2Unrolled = Theta2.toArray();
		double[] weightVector = concatenateArrays(Theta1Unrolled,
				Theta2Unrolled);

		return new DenseDoubleVector(weightVector);
	}

	private static double[] concatenateArrays(double[] a1, double[] a2) {
		double[] result = new double[a1.length + a2.length];
		System.arraycopy(a1, 0, result, 0, a1.length);
		System.arraycopy(a2, 0, result, a1.length, a2.length);

		return result;
	}

	public DoubleMatrix[] getThetas() {
		return Thetas;
	}
}
