package com.dvcs.tools;

import junit.framework.Assert;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.junit.Test;

public class MatlabMatrixFactoryTestCase {

	@Test
	public void testReshape() {
		double[] unrolled = new double[] { 0.0, 1.0, 2.0, 3.0 };
		RealMatrix expected = new Array2DRowRealMatrix(
				new double[][] { new double[] { 0.0, 1.0 },
						new double[] { 2.0, 3.0 }});
		
		RealMatrix reshaped = MatrixTools.reshape(unrolled);
		Assert.assertEquals(expected, reshaped);
	}
	
	@Test(expected=RuntimeException.class)
	public void testReshapeException() {
		MatrixTools.reshape(new double[] { 0.0, 1.0, 2.0 });
	}

}
