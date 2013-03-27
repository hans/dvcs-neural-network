package com.dvcs.neuralnetwork;

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