package com.dvcs.neuralnetwork.driver;

import com.dvcs.neuralnetwork.driver.Driver.OutputProvider;

public class DriverOutputManager implements OutputProvider {

	private double[] output;
	
	private static final int OUTPUT_SIZE = 3;
	private static final int OUTPUT_INDEX_LEFT_ARROW = 0;
	private static final int OUTPUT_INDEX_RIGHT_ARROW = 1;
	private static final int OUTPUT_INDEX_UP_ARROW = 2;
	
	public DriverOutputManager() {
		output = new double[OUTPUT_SIZE];
	}
	
	public void setLeftArrowEnabled(boolean l) {
		output[OUTPUT_INDEX_LEFT_ARROW] = l ? 1 : 0;
	}
	
	public void setRightArrowEnabled(boolean l) {
		output[OUTPUT_INDEX_RIGHT_ARROW] = l ? 1 : 0;
	}
	
	public void setUpArrowEnabled(boolean l) {
		output[OUTPUT_INDEX_UP_ARROW] = l ? 1 : 0;
	}
	
	@Override
	public double[] getOutput() {
		return output;
	}

}
