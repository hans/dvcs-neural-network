package com.dvcs.neuralnetwork;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import com.dvcs.neuralnetwork.NeuralNetwork.ForwardPropagationResult;
import com.dvcs.neuralnetwork.NeuralNetwork.WeightDeltas;
import com.dvcs.tools.MatrixTools;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.minimize.CostFunction;
import de.jungblut.math.tuple.Tuple;

public class NeuralNetworkCostFunction implements CostFunction {

	private NeuralNetwork nn;
	private DoubleMatrix x;
	private DoubleMatrix y;
	private double lambda;

	/**
	 * This weight matrix will begin with the same value as assigned in the
	 * provided neural network. It will be optimized and repeatedly fed back to
	 * the network (while the network's actual weights remain unmodified during
	 * the optimization process).
	 */
	private DoubleMatrix Theta1;

	/**
	 * This weight matrix will begin with the same value as assigned in the
	 * provided neural network. It will be optimized and repeatedly fed back to
	 * the network (while the network's actual weights remain unmodified during
	 * the optimization process).
	 */
	private DoubleMatrix Theta2;

	/**
	 * @param x
	 *            Matrix of examples (where each row represents an example and
	 *            each column represents a unit)
	 * @param y
	 *            Optimal output unit values (where each row is a unit and each
	 *            column is an example). In a network with desired binary output
	 *            values, the output matrix will consist of columns from an
	 *            identity matrix.
	 * @param lambda
	 *            Regularization parameter
	 */
	public NeuralNetworkCostFunction(NeuralNetwork nn, DoubleMatrix x,
			DoubleMatrix y, double lambda) {
		this.nn = nn;
		this.x = x;
		this.y = y;
		this.lambda = lambda;

		Theta1 = MatrixTools.copy(nn.Theta1);
		Theta2 = MatrixTools.copy(nn.Theta2);
	}

	/**
	 * Compute the cost and gradient for predicting with the given weights.
	 */
	@Override
	public Tuple<Double, DoubleVector> evaluateCost(DoubleVector point) {
		DoubleMatrix[] weights = nn.convertPointToWeightMatrices(point);
		DoubleMatrix Theta1 = weights[0];
		DoubleMatrix Theta2 = weights[1];

		ForwardPropagationResult fResult = nn.feedForward(x, Theta1, Theta2);
		WeightDeltas errorDeltas = nn.backpropagate(fResult, y, Theta1, Theta2);
		WeightDeltas regularizationDeltas = buildRegularizationDeltas();

		int m = x.getRows();

		DoubleMatrix[] gradMatrices = new DoubleMatrix[] {
				errorDeltas.getDelta1()
						.add(regularizationDeltas.getDelta1().mul(lambda))
						.mul(1.0 / m),

				errorDeltas.getDelta2()
						.add(regularizationDeltas.getDelta2().mul(lambda))
						.mul(1.0 / m) };

		DoubleVector gradVector = nn.convertWeightMatricesToPoint(gradMatrices);

		// Evaluate cost
		double cost = this.getCost(x, fResult.getA3(), y, lambda);
		
		return new Tuple<Double, DoubleVector>(cost, gradVector);
	}

	private WeightDeltas buildRegularizationDeltas() {
		DoubleMatrix Delta1 = MatrixTools.copy(Theta1);
		DoubleMatrix Delta2 = MatrixTools.copy(Theta2);

		// Ignore bias weights
		for (DoubleMatrix m : new DoubleMatrix[] { Delta1, Delta2 }) {
			for (int j = 0; j < m.getRows(); j++) {
				m.put(j, 0, 0);
			}
		}

		return nn.new WeightDeltas(Delta1, Delta2);
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
	private double getCost(DoubleMatrix x, DoubleMatrix outputLayer,
			DoubleMatrix y, double lambda) {
		/**
		 * Cost function: fitting
		 */

		int m = x.getRows();
		double fittingCost = 0;

		for (int i = 0; i < m; i++) {
			// Fetch a $y^{(i)}$ column vector for this example
			DoubleMatrix yi = y.getColumn(i);

			// Fetch the column vector $h^{(i)}$ from the output layer. Each row
			// of this vector represents a given class's output.
			DoubleMatrix hi = outputLayer.getColumn(i);

			double exampleCost = getClassificationCost(hi, yi);
			fittingCost += exampleCost;
		}

		/**
		 * Cost function: regularization
		 * 
		 * Ignore the first column of the weight matrices, which corresponds to
		 * the weights for the bias unit.
		 */

		DoubleMatrix theta1Shaved = Theta1.getRange(0, Theta1.getRows(), 1,
				Theta1.getColumns());
		DoubleMatrix theta2Shaved = Theta2.getRange(0, Theta2.getRows(), 1,
				Theta2.getColumns());
		double regularizationCost = MatrixTools.matrixSum(theta1Shaved
				.mul(theta1Shaved))
				+ MatrixTools.matrixSum(theta2Shaved.mul(theta2Shaved));

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
	private static double getClassificationCost(DoubleMatrix h, DoubleMatrix y) {
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
		DoubleMatrix positivePart = y.mul(MatrixFunctions.log(h));

		/**
		 * In the negative part, any units where `y` is 1 (i.e., where the unit
		 * is positive) are nulled out.
		 */
		DoubleMatrix negativePart = y.mul(-1).add(1)
				.mul(MatrixFunctions.log(h.mul(-1).add(1)));

		DoubleMatrix costPerClass = positivePart.add(negativePart).mul(-1);

		return costPerClass.sum();
	}

}
