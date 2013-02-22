package com.dvcs.neuralnetwork;

import org.jblas.DoubleMatrix;

import com.dvcs.tools.MatrixTools;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.dense.DenseDoubleVector;
import de.jungblut.math.minimize.Fmincg;

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
	 */
	public void train(DoubleMatrix x, DoubleMatrix y, double lambda) {
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
		double[] Theta1Vec = Theta1.toArray();
		double[] Theta2Vec = Theta2.toArray();
		double[] initParams = new double[Theta1Vec.length + Theta2Vec.length];
		System.arraycopy(Theta1Vec, 0, initParams, 0, Theta1Vec.length);
		System.arraycopy(Theta2Vec, 0, initParams, Theta1Vec.length,
				Theta2Vec.length);

		NeuralNetworkCostFunction cost = new NeuralNetworkCostFunction(this, x,
				y, lambda);
		DoubleVector parameters = Fmincg.minimizeFunction(cost,
				new DenseDoubleVector(initParams), 50, true);
		System.out.println(parameters);
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

		DoubleMatrix A2 = fResult.getA2();
		DoubleMatrix A3 = fResult.getA3();

		DoubleMatrix A2t = A2.transpose();
		DoubleMatrix A3t = A3.transpose();

		for (int i = 0; i < m; i++) {
			// The first "error" values are actual residuals.
			DoubleMatrix delta3 = A3.getColumn(i).sub(y.getColumn(i));
			
			/**
			 * POOF!
			 */

			// Propagate back to the hidden layer, using error values and
			// parameters (Theta) to build a weighted sum.
			DoubleMatrix delta2 = Theta2.transpose().mmul(delta3);
			// Remove the error corresponding to the bias unit and finish.
			delta2 = delta2.getRange(1, delta2.getRows(), 0,
					delta2.getColumns());
			
			/**
			 * Aaand.. we have our deltas.
			 */

			Delta1 = Delta1.add(delta2.mmul(A2t.getRow(i)));
			Delta2 = Delta2.add(delta3.mmul(A3t.getRow(i)));
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
}
