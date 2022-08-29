using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeatNetwork.Libraries;

namespace NeatNetwork.Groupings
{
    /// <summary>
    /// A connection is backward connected
    /// </summary>
    public class Connection
    {
        internal Range InputRange;
        internal Range ConnectedNetworkOutputRange;
        internal List<List<double>> weights;
        internal int ConnectedNetworkI;
    }
}
