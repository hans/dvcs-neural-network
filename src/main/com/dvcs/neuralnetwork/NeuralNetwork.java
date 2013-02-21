package com.dvcs.neuralnetwork;

import org.apache.commons.math3.analysis.function.Log;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import com.dvcs.tools.MatrixTools;

public class NeuralNetwork {

	final double RANDOM_WEIGHT_MATRIX_MINIMUM = 0;
	final double RANDOM_WEIGHT_MATRIX_MAXIMUM = 0.12;

	RealMatrix Theta1;
	RealMatrix Theta2;

	public NeuralNetwork(int numLayer1Units, int numLayer2Units,
			int numLayer3Units) {
		Theta1 = MatrixTools.randomMatrix(numLayer2Units, numLayer1Units + 1,
				RANDOM_WEIGHT_MATRIX_MINIMUM, RANDOM_WEIGHT_MATRIX_MAXIMUM);
		Theta2 = MatrixTools.randomMatrix(numLayer3Units, numLayer2Units + 1,
				RANDOM_WEIGHT_MATRIX_MINIMUM, RANDOM_WEIGHT_MATRIX_MAXIMUM);
	}

	public NeuralNetwork(RealMatrix t1, RealMatrix t2) {
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
	public void train(RealMatrix x, RealMatrix y, double lambda) {
		if (x.getRowDimension() != y.getColumnDimension()) {
			throw new RuntimeException(
					"Output matrix dimensions do not correspond with those of the example matrix");
		}

		ForwardPropagationResult fResult = feedForward(x);
		RealMatrix outputLayer = fResult.getA3();
		if (outputLayer.getRowDimension() != y.getRowDimension()
				|| outputLayer.getColumnDimension() != y.getColumnDimension()) {
			throw new RuntimeException(
					"Given output matrix dimensions do not correspond with those of the actual output matrix produced by the network");
		}

		double cost = getCost(x, outputLayer, y, lambda);

		// TODO run optimizer
	}

	public class Gradients {
		private RealMatrix Theta1Gradient;
		private RealMatrix Theta2Gradient;

		public Gradients(RealMatrix theta1Gradient, RealMatrix theta2Gradient) {
			Theta1Gradient = theta1Gradient;
			Theta2Gradient = theta2Gradient;
		}

		public RealMatrix getTheta1Gradient() {
			return Theta1Gradient;
		}

		public RealMatrix getTheta2Gradient() {
			return Theta2Gradient;
		}
	}

	public Gradients buildGradients(ForwardPropagationResult fResult,
			RealMatrix y, double lambda) {
		int m = y.getColumnDimension();

		WeightDeltas errorDeltas = backpropagate(fResult, y);
		WeightDeltas regularizationDeltas = buildRegularizationDeltas();

		RealMatrix Theta1Grad = errorDeltas.getDelta1()
				.add(regularizationDeltas.getDelta1().scalarMultiply(lambda))
				.scalarMultiply(1 / m);
		RealMatrix Theta2Grad = errorDeltas.getDelta2()
				.add(regularizationDeltas.getDelta2().scalarMultiply(lambda))
				.scalarMultiply(1 / m);

		return new Gradients(Theta1Grad, Theta2Grad);
	}

	public class WeightDeltas {
		private RealMatrix Delta1;
		private RealMatrix Delta2;

		public WeightDeltas(RealMatrix delta1, RealMatrix delta2) {
			Delta1 = delta1;
			Delta2 = delta2;
		}

		public RealMatrix getDelta1() {
			return Delta1;
		}

		public RealMatrix getDelta2() {
			return Delta2;
		}
	}

	/**
	 * Determine what changes should be made to the weight matrices to lower
	 * cost on the given data set.
	 */
	private WeightDeltas backpropagate(ForwardPropagationResult fResult,
			RealMatrix y) {
		int m = y.getColumnDimension();

		RealMatrix Delta1 = new Array2DRowRealMatrix(Theta1.getRowDimension(),
				Theta1.getColumnDimension());
		RealMatrix Delta2 = new Array2DRowRealMatrix(Theta2.getRowDimension(),
				Theta2.getColumnDimension());

		RealMatrix A1 = fResult.getA1();
		RealMatrix A2 = fResult.getA2();
		RealMatrix A3 = fResult.getA3();

		RealMatrix A2t = A2.transpose();
		RealMatrix A3t = A3.transpose();

		for (int i = 0; i < m; i++) {
			// The first "error" values are actual residuals.
			RealVector delta3Vector = A3.getColumnVector(i).subtract(
					y.getColumnVector(i));
			RealMatrix delta3 = new Array2DRowRealMatrix(delta3Vector.toArray());

			// Propagate back to the hidden layer, using error values and
			// parameters (Theta) to build a weighted sum.
			RealMatrix delta2 = Theta2.transpose().multiply(delta3);
			// Remove the error corresponding to the bias unit and finish.
			delta2 = delta2.getSubMatrix(1, delta2.getRowDimension(), 0,
					delta2.getColumnDimension());

			Delta1 = Delta1.add(delta2.multiply(A2t.getRowMatrix(i)));
			Delta2 = Delta2.add(delta3.multiply(A3t.getRowMatrix(i)));
		}

		return new WeightDeltas(Delta1, Delta2);
	}

	private WeightDeltas buildRegularizationDeltas() {
		RealMatrix Delta1 = Theta1.copy();
		RealMatrix Delta2 = Theta2.copy();

		// Ignore bias weights
		for (RealMatrix m : new RealMatrix[] { Delta1, Delta2 }) {
			for (int j = 0; j < m.getRowDimension(); j++) {
				m.setEntry(j, 0, 0);
			}
		}

		return new WeightDeltas(Delta1, Delta2);
	}

	/**
	 * Determine the cost of a prediction given the actual (i.e., expected)
	 * output values.
	 * 
	 * @param x
	 *            Matrix of examples (where each row represents an example and
	 *            each column represents a unit)
	 * @param outputLayer
	 *            A matrix representing the output layer of the network, where
	 *            each column represents an example and each row represents a
	 *            unit in the output layer
	 * @param y
	 *            A matrix of the optimal output unit value, where each row is a
	 *            unit in the output layer and each column is an example
	 * @param lambda
	 *            Regularization parameter
	 */
	private double getCost(RealMatrix x, RealMatrix outputLayer, RealMatrix y,
			double lambda) {
		/**
		 * Cost function: fitting
		 */

		int m = x.getRowDimension();
		double fittingCost = 0;

		for (int i = 0; i < m; i++) {
			// Fetch a $y^{(i)}$ column vector for this example
			RealVector yi = y.getColumnVector(i);

			// Fetch the column vector $h^{(i)}$ from the output layer. Each row
			// of this vector represents a given class's output.
			RealVector hi = outputLayer.getColumnVector(i);

			double exampleCost = getClassificationCost(hi, yi);
			fittingCost += exampleCost;
		}

		/**
		 * Cost function: regularization
		 * 
		 * Ignore the first column of the weight matrices, which corresponds to
		 * the weights for the bias unit.
		 */

		RealMatrix theta1Shaved = Theta1.getSubMatrix(0,
				Theta1.getRowDimension(), 1, Theta1.getRowDimension());
		RealMatrix theta2Shaved = Theta2.getSubMatrix(0,
				Theta2.getRowDimension(), 1, Theta2.getRowDimension());
		double regularizationCost = MatrixTools.matrixSum(MatrixTools.squareMatrixElements(theta1Shaved))
				+ MatrixTools.matrixSum(MatrixTools.squareMatrixElements(theta2Shaved));

		double totalCost = (fittingCost + lambda / 2 * regularizationCost) / m;
		return totalCost;
	}

	/**
	 * Determine the cost of classification for a single example using this
	 * network.
	 * 
	 * @param h
	 *            A column vector of prediction values where each row represents
	 *            an output unit
	 * @param y
	 *            A column vector of actual output values (with the same number
	 *            of components as `h`)
	 * @return A column vector where each row represents the classification cost
	 *         given the error for the corresponding unit
	 */
	@SuppressWarnings("deprecation")
	private static double getClassificationCost(RealVector h, RealVector y) {
		/**
		 * The classification cost formula assumes that `y` is a column of the
		 * identity matrix: that is, one of its rows has a 1 and the rest have
		 * 0.
		 * 
		 * The formula has two parts: the first which accounts for positive
		 * examples (i.e., where a `y` cell value is 1) and negative examples
		 * (i.e., where a `y` cell value is 0).
		 */

		/**
		 * The positive part accounts for positive examples (where the `y` cell
		 * is 1). Any units where `y` is 0 have a zero value in the positive
		 * part.
		 */
		RealVector positivePart = y.ebeMultiply(h.map(new Log()));

		/**
		 * In the negative part, any units where `y` is 1 (i.e., where the unit
		 * is positive) are nulled out.
		 */
		RealVector negativePart = y.mapMultiply(-1).mapAdd(1)
				.ebeMultiply(h.mapMultiply(-1).mapAdd(1).map(new Log()));

		RealVector costPerClass = positivePart.add(negativePart)
				.mapMultiply(-1);

		double totalCost = 0;
		for (int i = 0; i < costPerClass.getDimension(); i++) {
			totalCost += costPerClass.getEntry(i);
		}

		return totalCost;
	}

	public class ForwardPropagationResult {
		private RealMatrix a1;
		private RealMatrix z2;
		private RealMatrix a2;
		private RealMatrix z3;
		private RealMatrix a3;

		public ForwardPropagationResult(RealMatrix a1, RealMatrix z2,
				RealMatrix a2, RealMatrix z3, RealMatrix a3) {
			this.a1 = a1;
			this.z2 = z2;
			this.a2 = a2;
			this.z3 = z3;
			this.a3 = a3;
		}

		public RealMatrix getA1() {
			return a1;
		}

		public RealMatrix getZ2() {
			return z2;
		}

		public RealMatrix getA2() {
			return a2;
		}

		public RealMatrix getZ3() {
			return z3;
		}

		public RealMatrix getA3() {
			return a3;
		}
	}

	/**
	 * Feed a collection of examples forward through the network.
	 * 
	 * @param x
	 *            Example matrix, where each row represents an example and each
	 *            column represents a unit
	 * @return A matrix of output layer units, where each row represents a unit
	 *         and each column represents an example
	 */
	public ForwardPropagationResult feedForward(RealMatrix x) {
		RealMatrix a1 = addBiasUnit(x.transpose());

		RealMatrix z2 = Theta1.multiply(a1);
		RealMatrix a2 = MatrixTools.matrixSigmoid(addBiasUnit(z2));

		RealMatrix z3 = Theta2.multiply(a2);
		RealMatrix a3 = MatrixTools.matrixSigmoid(z3);

		return new ForwardPropagationResult(a1, z2, a2, z3, a3);
	}

	/**
	 * Given a matrix of examples (where each row represents a unit and each
	 * column represents an example), return an array where each element
	 * corresponds to an example and its value is the index of the class whose
	 * corresponding output unit has the highest value.
	 */
	public int[] predict(RealMatrix a3) {
		int[] results = new int[a3.getColumnDimension()];

		for (int j = 0; j < a3.getColumnDimension(); j++) {
			results[j] = maxIndex(a3.getColumn(j));
		}

		return results;
	}

	/**
	 * Return the index of the element in the array with the maximum value.
	 */
	static int maxIndex(double[] xs) {
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
	static RealMatrix addBiasUnit(RealMatrix a) {
		RealMatrix ret = new Array2DRowRealMatrix(a.getRowDimension() + 1,
				a.getColumnDimension());

		// Initialize the bias unit (first row)
		for (int j = 0; j < ret.getColumnDimension(); j++) {
			ret.setEntry(0, j, 1);
		}

		// Place the original layer data below this first row
		ret.setSubMatrix(a.getData(), 1, 0);

		return ret;
	}
}
