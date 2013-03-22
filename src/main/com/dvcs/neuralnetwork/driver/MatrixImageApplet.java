package com.dvcs.neuralnetwork.driver;

import java.awt.Color;
import java.awt.Graphics;

import javax.swing.JApplet;

import org.jblas.DoubleMatrix;

public class MatrixImageApplet extends JApplet {

	DoubleMatrix image;

	static final int SCALE = 20;

	public void setImage(DoubleMatrix i) {
		image = i.transpose();
		setSize(image.getColumns() * SCALE, image.getRows() * SCALE);

		repaint();
	}

	@Override
	public void paint(Graphics g) {
		for (int i = 0; i < image.getRows(); i++) {
			for (int j = 0; j < image.getColumns(); j++) {
				float intensity = (float) image.get(i, j);
				Color c = new Color(intensity, intensity, intensity);

				g.setColor(c);
				g.fillRect(i * SCALE, j * SCALE, SCALE, SCALE);
			}
		}
	}

}
