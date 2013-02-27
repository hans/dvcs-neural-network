package com.dvcs.neuralnetwork;

import java.util.Arrays;

import junit.framework.Assert;

import org.jblas.DoubleMatrix;
import org.junit.Before;
import org.junit.Test;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.dense.DenseDoubleVector;
import de.jungblut.math.minimize.CostFunction;
import de.jungblut.math.tuple.Tuple;

/**
 * Creates a small neural network to check the backpropagation gradients. It
 * will output the analytical gradients produced by your backprop code and the
 * numerical gradients. These two gradient computations should result in very
 * similar values.
 */
public class NeuralNetworkGradientTestCase {

	static final double LAMBDA = 0;

	static final int INPUT_LAYER_SIZE = 3;
	static final int HIDDEN_LAYER_SIZE = 5;
	static final int NUM_LABELS = 3;
	static final int M = 5;

	static final double NUMERICAL_GRADIENT_SHIFT = 1e-4;

	NeuralNetwork nn;
	DoubleMatrix Theta1;
	DoubleMatrix Theta2;

	DoubleMatrix X;
	DoubleMatrix Y;

	@Before
	public void setUp() throws Exception {
		Theta1 = initializeWeights(HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE);
		Theta2 = initializeWeights(NUM_LABELS, HIDDEN_LAYER_SIZE);

		X = initializeWeights(M, INPUT_LAYER_SIZE - 1);

		DoubleMatrix y = new DoubleMatrix(M, 1);
		for (int i = 0; i < M; i++) {
			y.put(i, 0, (i + 1) % NUM_LABELS + 1);
		}

		Y = NeuralNetwork.buildYMatrix(y, NUM_LABELS);

		nn = new NeuralNetwork(Theta1, Theta2);
	}

	@Test
	public void test() {
		NeuralNetworkCostFunction costFunction = new NeuralNetworkCostFunction(
				nn, X, Y, LAMBDA);
		DoubleVector params = NeuralNetwork
				.convertWeightMatricesToPoint(new DoubleMatrix[] { Theta1,
						Theta2 });

		Tuple<Double, DoubleVector> result = costFunction.evaluateCost(params);
		
		DoubleVector resultVector = result.getSecond();
		DoubleVector numericalGradient = computeNumericalGradient(costFunction,
				params);

		DoubleVector topDiff = numericalGradient.subtract(resultVector);
		DoubleVector botDiff = numericalGradient.add(resultVector);

		double rating = Math.sqrt(topDiff.dot(topDiff))
				/ Math.sqrt(botDiff.dot(botDiff));
		System.out.println(rating);

		Assert.assertTrue(
				"Numerical gradient and backpropagated gradient are very close",
				rating < 1e-9);
	}

	DoubleVector computeNumericalGradient(CostFunction costFunction,
			DoubleVector theta) {
		DoubleVector perturb = new DenseDoubleVector(theta.getLength(), 0);
		DoubleVector grad = new DenseDoubleVector(theta.getLength(), 0);

		for (int p = 0; p < theta.getLength(); p++) {
			// Set perturbation vector
			perturb.set(p, NUMERICAL_GRADIENT_SHIFT);

			Tuple<Double, DoubleVector> cost1 = costFunction.evaluateCost(theta
					.add(perturb));
			Tuple<Double, DoubleVector> cost2 = costFunction.evaluateCost(theta
					.subtract(perturb));

			grad.set(p, (cost1.getFirst() - cost2.getFirst())
					/ (2.0 * NUMERICAL_GRADIENT_SHIFT));

			// Reset perturbation vector
			perturb.set(p, 0);
		}

		return grad;
	}

	/**
	 * Initialize the weights of a layer with `fan_in` incoming connections and
	 * `fan_out` outgoing connections using a fixed strategy.
	 * 
	 * @param fan_out
	 * @param fan_in
	 * @return A matrix with `1 + fan_in` rows and `fan_out` columns (the first
	 *         row handles the "bias" terms)
	 */
	static DoubleMatrix initializeWeights(int fan_out, int fan_in) {
		DoubleMatrix ret = new DoubleMatrix(fan_out, 1 + fan_in);

		for (int j = 0; j < ret.getColumns(); j++) {
			for (int i = 0; i < ret.getRows(); i++) {
				ret.put(i, j, Math.sin(j * ret.getRows() + i + 1) / 10.0);
			}
		}

		return ret;
	}

}
