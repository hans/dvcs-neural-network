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

	DoubleMatrix Theta1;
	DoubleMatrix Theta2;

	public NeuralNetwork(int numLayer1Units, int numLayer2Units,
			int numLayer3Units) {
		Theta1 = DoubleMatrix
				.rand(numLayer2Units, numLayer1Units + 1)
				.mul(RANDOM_WEIGHT_MATRIX_MAXIMUM
						- RANDOM_WEIGHT_MATRIX_MINIMUM)
				.add(RANDOM_WEIGHT_MATRIX_MINIMUM);

		Theta2 = DoubleMatrix
				.rand(numLayer3Units, numLayer2Units + 1)
				.mul(RANDOM_WEIGHT_MATRIX_MAXIMUM
						- RANDOM_WEIGHT_MATRIX_MINIMUM)
				.add(RANDOM_WEIGHT_MATRIX_MINIMUM);
	}

	public NeuralNetwork(DoubleMatrix t1, DoubleMatrix t2) {
		Theta1 = t1;
		Theta2 = t2;
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
			double lambda, MinimizerListener listener) {
		DoubleMatrix Y = buildYMatrix(yVector, k);
		train(x, Y, lambda, listener);
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
	 */
	public void train(DoubleMatrix x, DoubleMatrix y, double lambda,
			MinimizerListener listener) {
		if (x.getRows() != y.getColumns()) {
			throw new RuntimeException(
					"Output matrix ss do not correspond with those of the example matrix");
		}

		ForwardPropagationResult fResult = feedForward(x);
		DoubleMatrix outputLayer = fResult.getA3();
		if (outputLayer.getRows() != y.getRows()
				|| outputLayer.getColumns() != y.getColumns()) {
			throw new RuntimeException(
					"Given output matrix ss do not correspond with those of the actual output matrix produced by the network");
		}

		// Unroll Theta1 and Theta2, then concatenate. This forms our initial
		// parameter vector.
		DoubleVector initParams = convertWeightMatricesToPoint(new DoubleMatrix[] {
				Theta1, Theta2 });

		NeuralNetworkCostFunction cost = new NeuralNetworkCostFunction(this, x,
				y, lambda);
		DoubleVector parameters = Fmincg.minimizeFunction(cost, initParams, 50,
				listener);

		DoubleMatrix[] weights = convertPointToWeightMatrices(parameters);
		Theta1 = weights[0];
		Theta2 = weights[1];
	}

	public class WeightDeltas {
		private DoubleMatrix Delta1;
		private DoubleMatrix Delta2;

		public WeightDeltas(DoubleMatrix delta1, DoubleMatrix delta2) {
			Delta1 = delta1;
			Delta2 = delta2;
		}

		public DoubleMatrix getDelta1() {
			return Delta1;
		}

		public DoubleMatrix getDelta2() {
			return Delta2;
		}
	}

	public WeightDeltas backpropagate(ForwardPropagationResult fResult,
			DoubleMatrix y) {
		return backpropagate(fResult, y, Theta1, Theta2);
	}

	/**
	 * Determine what changes should be made to the weight matrices to lower
	 * cost on the given data set.
	 */
	public WeightDeltas backpropagate(ForwardPropagationResult fResult,
			DoubleMatrix y, DoubleMatrix Theta1, DoubleMatrix Theta2) {
		int m = y.getColumns();

		DoubleMatrix Delta1 = new DoubleMatrix(Theta1.getRows(),
				Theta1.getColumns());
		DoubleMatrix Delta2 = new DoubleMatrix(Theta2.getRows(),
				Theta2.getColumns());

		DoubleMatrix A1 = fResult.getA1();
		DoubleMatrix A2 = fResult.getA2();
		DoubleMatrix A3 = fResult.getA3();
		DoubleMatrix Z2 = fResult.getZ2();

		for (int i = 0; i < m; i++) {
			// The first "error" values are actual residuals.
			DoubleMatrix delta3 = A3.getColumn(i).sub(y.getColumn(i));

			// Propagate back to the hidden layer, using error values and
			// parameters (Theta) to build a weighted sum.
			DoubleMatrix delta2 = Theta2.transpose().mmul(delta3);
			// Remove the error corresponding to the bias unit and finish.
			delta2 = delta2.getRange(1, delta2.getRows(), 0,
					delta2.getColumns()).mul(
					MatrixTools.matrixSigmoidGradient(Z2.getColumn(i)));

			DoubleMatrix Delta1Delta = delta2.mmul(A1.getColumn(i).transpose());
			DoubleMatrix Delta2Delta = delta3.mmul(A2.getColumn(i).transpose());

			Delta1 = Delta1.add(Delta1Delta);
			Delta2 = Delta2.add(Delta2Delta);
		}

		return new WeightDeltas(Delta1, Delta2);
	}

	public class ForwardPropagationResult {
		private DoubleMatrix a1;
		private DoubleMatrix z2;
		private DoubleMatrix a2;
		private DoubleMatrix z3;
		private DoubleMatrix a3;

		public ForwardPropagationResult(DoubleMatrix a1, DoubleMatrix z2,
				DoubleMatrix a2, DoubleMatrix z3, DoubleMatrix a3) {
			this.a1 = a1;
			this.z2 = z2;
			this.a2 = a2;
			this.z3 = z3;
			this.a3 = a3;
		}

		public DoubleMatrix getA1() {
			return a1;
		}

		public DoubleMatrix getZ2() {
			return z2;
		}

		public DoubleMatrix getA2() {
			return a2;
		}

		public DoubleMatrix getZ3() {
			return z3;
		}

		public DoubleMatrix getA3() {
			return a3;
		}
	}

	public ForwardPropagationResult feedForward(DoubleMatrix x) {
		return feedForward(x, Theta1, Theta2);
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
			DoubleMatrix Theta1, DoubleMatrix Theta2) {
		DoubleMatrix a1 = addBiasUnit(x.transpose());

		DoubleMatrix z2 = Theta1.mmul(a1);
		DoubleMatrix a2 = MatrixTools.matrixSigmoid(addBiasUnit(z2));

		DoubleMatrix z3 = Theta2.mmul(a2);
		DoubleMatrix a3 = MatrixTools.matrixSigmoid(z3);

		return new ForwardPropagationResult(a1, z2, a2, z3, a3);
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
		return convertPointToWeightMatrices(point, Theta1.getRows(),
				Theta1.getColumns(), Theta2.getRows(), Theta2.getColumns());
	}

	/**
	 * A "point" provided by the optimization algorithm consists of a series of
	 * structures, each of which contains a single parameter value. We can turn
	 * this into a flat list of parameters.
	 * 
	 * Reshape the flat parameter list required by the optimization method into
	 * the two separate matrices that it represents (i.e., Theta1 and Theta2
	 * unrolled and then concatenated).
	 */
	static DoubleMatrix[] convertPointToWeightMatrices(DoubleVector point,
			int Theta1Rows, int Theta1Cols, int Theta2Rows, int Theta2Cols) {
		double[] params = point.toArray();

		int Theta1Length = Theta1Rows * Theta1Cols;
		double[] Theta1Unrolled = Arrays.copyOfRange(params, 0, Theta1Length);
		DoubleMatrix Theta1 = MatrixTools.reshape(Theta1Unrolled, Theta1Rows,
				Theta1Cols);

		int Theta2Length = Theta2Rows * Theta2Cols;
		double[] Theta2Unrolled = Arrays.copyOfRange(params, Theta1Length,
				Theta1Length + Theta2Length);
		DoubleMatrix Theta2 = MatrixTools.reshape(Theta2Unrolled, Theta2Rows,
				Theta2Cols);

		return new DoubleMatrix[] { Theta1, Theta2 };
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
}
