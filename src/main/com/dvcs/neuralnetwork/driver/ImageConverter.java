package com.dvcs.neuralnetwork.driver;

import java.awt.image.BufferedImage;

import org.jblas.DoubleMatrix;

import com.dvcs.tools.MatrixTools;

public class ImageConverter {

	public static DoubleMatrix convertImageToMatrix(BufferedImage image,
			boolean makeGrayscale) {
		int m = image.getWidth(null);
		int n = image.getHeight(null);

		if (makeGrayscale) {
			BufferedImage newImage = new BufferedImage(m, n,
					BufferedImage.TYPE_BYTE_GRAY);
			newImage.getGraphics().drawImage(image, 0, 0, null);
			image = newImage;
		}

		DoubleMatrix ret = new DoubleMatrix(m, n);
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				ret.put(i, j, image.getRGB(i, j));
			}
		}
		
		return ret;
	}
	
	/**
	 * Normalize all elements of a matrix such that they lie in the range from 0
	 * to 1 (inclusive on both ends).
	 */
	public static DoubleMatrix normalize(DoubleMatrix m) {
		m = MatrixTools.copy(m);

		double max = Double.MIN_VALUE;
		double min = Double.MAX_VALUE;

		for ( int i = 0; i < m.getRows(); i++ ) {
			for ( int j = 0; j < m.getColumns(); j++ ) {
				double value = m.get(i, j);

				if ( value > max ) {
					max = value;
				} else if ( value < min ) {
					min = value;
				}
			}
		}

		for ( int i = 0; i < m.getRows(); i++ ) {
			for ( int j = 0; j < m.getColumns(); j++ ) {
				double value = m.get(i, j);
				value = (value - min) / (max - min);
				if ( value < 0 ) {
					System.out.println("--- fail");
				}

				m.put(i, j, value);
			}
		}

		return m;
	}

}
