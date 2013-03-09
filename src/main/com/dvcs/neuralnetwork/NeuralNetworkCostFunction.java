package com.dvcs.neuralnetwork;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import com.dvcs.neuralnetwork.NeuralNetwork.ForwardPropagationResult;
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
	 * These weight matrices will begin with the same value as assigned in the
	 * provided neural network. They will be optimized and repeatedly fed back
	 * to the network (while the network's actual weights remain unmodified
	 * during the optimization process).
	 */
	private DoubleMatrix[] Thetas;

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

		Thetas = new DoubleMatrix[nn.Thetas.length];
		for (int i = 0; i < Thetas.length; i++) {
			Thetas[i] = MatrixTools.copy(nn.Thetas[i]);
		}
	}

	/**
	 * Compute the cost and gradient for predicting with the given weights.
	 */
	@Override
	public Tuple<Double, DoubleVector> evaluateCost(DoubleVector point) {
		DoubleMatrix[] Thetas = nn.convertPointToWeightMatrices(point);

		ForwardPropagationResult fResult = nn.feedForward(x, Thetas);
		DoubleMatrix[] errorDeltas = nn.backpropagate(fResult, y, Thetas);
		DoubleMatrix[] regularizationDeltas = buildRegularizationDeltas(Thetas);

		int m = x.getRows();

		DoubleMatrix[] gradMatrices = new DoubleMatrix[Thetas.length];
		for (int i = 0; i < Thetas.length; i++) {
			gradMatrices[i] = errorDeltas[i].add(
					regularizationDeltas[i].mul(lambda)).mul(1.0 / m);
		}

		DoubleVector gradVector = nn.convertWeightMatricesToPoint(gradMatrices);

		// Evaluate cost
		double cost = this.getCost(x, Thetas, fResult.getOutputLayer(), y, lambda);

		return new Tuple<Double, DoubleVector>(cost, gradVector);
	}

	private DoubleMatrix[] buildRegularizationDeltas(DoubleMatrix[] Thetas) {
		DoubleMatrix[] Deltas = new DoubleMatrix[Thetas.length];

		for (int t = 0; t < Thetas.length; t++) {
			// Ignore bias weights
			DoubleMatrix Delta = MatrixTools.copy(Thetas[t]);
			
			for (int i = 0; i < Delta.getRows(); i++) {
				Delta.put(i, 0, 0);
			}
			
			Deltas[t] = Delta;
		}

		return Deltas;
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
	public double getCost(DoubleMatrix x, DoubleMatrix[] Thetas,
			DoubleMatrix outputLayer, DoubleMatrix y, double lambda) {
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

		double regularizationCost = 0;
		for (int i = 0; i < Thetas.length; i++) {
			// Ignore the first column of the weight matrix, which corresponds
			// to the weights for the bias unit.
			DoubleMatrix ThetaShaved = Thetas[i].getRange(0, Thetas[i].getRows(), 1, Thetas[i].getColumns());
			
			regularizationCost += MatrixTools.matrixSum(ThetaShaved.mul(ThetaShaved));
		}

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
