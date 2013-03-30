package com.dvcs.neuralnetwork;

/**
 * Represents a single example to be learned.
 */
public class Example {
	private float[] x;
	private float[] y;

	public Example(float[] x, float[] y) {
		this.x = x;
		this.y = y;
	}

	public float[] getX() {
		return x;
	}

	public float[] getY() {
		return y;
	}
}