package com.dvcs.tools;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;

import javax.swing.JFrame;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import com.dvcs.handwriting.HandwritingExampleApplet;

public class MatlabMatrixFactory {
	public static RealMatrix loadFromReader(Reader r) throws IOException,
			RuntimeException {
		RealMatrix ret = null;
		BufferedReader br = new BufferedReader(r);

		// Track matrix info that should be at top of file
		boolean readyForData = false;
		int rows = -1, cols = -1, curRow = 0;

		String line;

		while ((line = br.readLine()) != null) {
			if (line.startsWith("#")) {
				if (line.startsWith("# rows: ")) {
					rows = Integer
							.parseInt(line.substring("# rows: ".length()));
				} else if (line.startsWith("# columns: ")) {
					cols = Integer.parseInt(line.substring("# columns: "
							.length()));
				}

				readyForData = rows > -1 && cols > -1;
			} else {
				if (!readyForData)
					throw new RuntimeException(
							"Given file is missing Matlab text header");

				if (ret == null)
					ret = new Array2DRowRealMatrix(rows, cols);

				if (line.length() == 0) {
					continue;
				}

				String[] rowCells = line.split(" ");

				// Check dimensions (accont for extra "" cell in rowCells[0])
				if (cols != rowCells.length - 1)
					throw new RuntimeException("Row " + curRow
							+ " has the wrong number of cells (expected "
							+ cols + " but saw " + (rowCells.length - 1) + ")");

				// First cell is an empty string
				for (int i = 1; i <= cols; i++) {
					ret.setEntry(curRow, i - 1, Double.parseDouble(rowCells[i]));
				}

				curRow++;
			}
		}

		return ret;
	}

	/**
	 * Unroll a vector into a square matrix. The vector must have a length that
	 * is a perfect square.
	 */
	public static RealMatrix reshape(double[] row) {
		double sq = Math.sqrt(row.length);
		if ((int) sq != sq) {
			throw new RuntimeException(
					"Reshape method provided with a vector whose length is not a square number");
		}
		
		int length = (int) sq;
		RealMatrix ret = new Array2DRowRealMatrix(length, length);
		
		for ( int i = 0; i < length; i++ ) {
			for ( int j = 0; j < length; j++ ) {
				ret.setEntry(i, j, row[i * length + j]);
			}
		}
		
		return ret;
	}
}
