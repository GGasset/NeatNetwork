using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeatNetwork.Groupings
{
    internal class AgroupatedNetwork
    {
        internal List<double> input;
        internal List<Connection> connections;
        public RNN n;

        public List<int> networksConnectedToThis { get; private set; }


        internal void ClearInput() => input = new List<double>(new double[n.InputLength]);
    }
}
