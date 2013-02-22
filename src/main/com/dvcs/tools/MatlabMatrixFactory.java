package com.dvcs.tools;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;

import org.jblas.DoubleMatrix;

public class MatlabMatrixFactory {
	public static DoubleMatrix loadFromReader(Reader r) throws IOException,
			RuntimeException {
		DoubleMatrix ret = null;
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
					ret = new DoubleMatrix(rows, cols);

				if (line.length() == 0) {
					continue;
				}

				String[] rowCells = line.split(" ");

				// Check dimensions (account for extra "" cell in rowCells[0])
				if (cols != rowCells.length - 1)
					throw new RuntimeException("Row " + curRow
							+ " has the wrong number of cells (expected "
							+ cols + " but saw " + (rowCells.length - 1) + ")");

				// First cell is an empty string
				for (int i = 1; i <= cols; i++) {
					ret.put(curRow, i - 1, Double.parseDouble(rowCells[i]));
				}

				curRow++;
			}
		}

		return ret;
	}
}
