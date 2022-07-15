using System;
using System.Drawing;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeatNetwork.NetworkFiles
{
    public class GradientValues
    {
        internal List<Point> previousActivationGradientsPosition;
        internal List<double> previousActivationGradients;
        internal List<double> weightGradients;
        internal double biasGradient;

        public GradientValues()
        {
            previousActivationGradientsPosition = new List<Point>();
            previousActivationGradients = new List<double>();
            weightGradients = new List<double>();
        }
    }
}
