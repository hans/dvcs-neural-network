package de.jungblut.math.minimize;

import de.jungblut.math.DoubleVector;

public interface MinimizerListener {

	/**
	 * Called when the minimizer has finished an iteration.
	 * 
	 * @param iterationNumber
	 * @param cost
	 * @param parameters
	 *            The newly adjusted parameters (after the most recently
	 *            finished iteration)
	 */
	public void minimizationIterationFinished(int iterationNumber, double cost,
			DoubleVector parameters);

}
