using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeatNetwork.Libraries
{
    public static class DataManipulation
    {
        private static int randomI = 0;

        public static (List<double[]> shuffledX, List<double[]> shuffledY) ShuffleData(List<double[]> X, List<double[]> y)
        {
            var shuffledX = new List<double[]>(X.ToArray());
            var shuffledY = new List<double[]>(y.ToArray());

            Random r = new Random(DateTime.Now.Millisecond + randomI++);
            for (int i = 0; i < shuffledX.Count; i++)
            {
                int interchangeI = r.Next(X.Count);
                (double[] currentX, double[] currentY) = (shuffledX[i], shuffledY[i]);
                (shuffledX[i], shuffledY[i]) = (shuffledX[interchangeI], shuffledY[interchangeI]);
                (shuffledX[interchangeI], shuffledY[interchangeI]) = (currentX, currentY);
            }

            return (shuffledX, shuffledY);
        }

        public static (List<List<double[]>> shuffledX, List<List<double[]>> shuffledY) ShuffleData(List<List<double[]>> X, List<List<double[]>> y)
        {
            var shuffledX = new List<List<double[]>>(X.ToArray());
            var shuffledY = new List<List<double[]>>(y.ToArray());

            Random r = new Random(DateTime.Now.Millisecond + randomI++);
            for (int i = 0; i < shuffledX.Count; i++)
            {
                int interchangeI = r.Next(X.Count);
                (List<double[]> currentX, List<double[]> currentY) = (shuffledX[i], shuffledY[i]);
                (shuffledX[i], shuffledY[i]) = (shuffledX[interchangeI], shuffledY[interchangeI]);
                (shuffledX[interchangeI], shuffledY[interchangeI]) = (currentX, currentY);
            }

            return (shuffledX, shuffledY);
        }

        public static ((List<double[]> firstX, List<double[]> firstY), (List<double[]> secondX, List<double[]> secondY)) SliceData(List<double[]> X, List<double[]> y, double slicePos = 0.8)
        {
            slicePos /= 1 + (X.Count - 1) * Convert.ToInt32(slicePos > 1);

            int dataPartitionI = Convert.ToInt32(X.Count * slicePos);
            List<double[]> firstX, firstY;
            firstX =  new List<double[]>();
            firstY = new List<double[]>();
            for (int i = 0; i < dataPartitionI; i++)
            {
                firstX.Add(X[i]);
                firstY.Add(y[i]);
            }

            List<double[]> secondX, secondY;
            secondX = new List<double[]>();
            secondY = new List<double[]>();
            for (int i = dataPartitionI; i < X.Count; i++)
            {
                secondX.Add(X[i]);
                secondY.Add(y[i]);
            }
            return ((firstX, firstY), (secondX, secondY));
        }
    }
}
