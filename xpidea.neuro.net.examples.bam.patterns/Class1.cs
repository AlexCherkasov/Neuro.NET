/*

  Neuro.NET - Library of neural networks for .NET
  Copyright (C) 2001-2015  Alex Cherkasov. All rights reserved.
                           email: info@xpidea.com
                           web: http://xpidea.com/

  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 2
  of the License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
  02110-1301, USA.

{   CREDITS:                                                    }
{   This work is based on publications of:                      }
{          -Christopher M. Bishop                               }
{          -Jose C. Principe                                    }
{          -Samuel J. Rogers                                    }
{          -Laurene V. Fausett                                  }
{          -Simon S. Haykin                                     }
  
*/

using System;
using xpidea.neuro.net.bam;
using xpidea.neuro.net.patterns;

namespace xpidea.neuro.net.examples.bam.patterns
{
    internal class Class1
    {
        private const int TrainingSets = 4;

        private static void SetPattern(Pattern aPattern, double i1, double i2, double i3, double i4, double o1,
            double o2, double o3, double o4)
        {
            aPattern.Input[0] = i1;
            aPattern.Input[1] = i2;
            aPattern.Input[2] = i3;
            aPattern.Input[3] = i4;
            aPattern.Output[0] = o1;
            aPattern.Output[1] = o2;
            aPattern.Output[2] = o3;
            aPattern.Output[3] = o4;
        }

        [STAThread]
        private static void Main(string[] args)
        {
            BidirectionalAssociativeMemorySystem BAMSystem;
            var patterns = new PatternsCollection(4, 4, 4);

            SetPattern(patterns[0], -1, 1, -1, 1, 1, -1, 1, -1); //invert
            SetPattern(patterns[1], -1, -1, 1, 1, 1, 1, -1, -1);
            SetPattern(patterns[2], 1, 1, -1, -1, -1, -1, 1, 1);
            SetPattern(patterns[3], -1, -1, -1, -1, 1, 1, 1, 1);

            BAMSystem = new BidirectionalAssociativeMemorySystem(4, 4);
            BAMSystem.Train(patterns);
            BAMSystem.SaveToFile("test.net");

            //We didn't exposed following pattens to BAM, but we'd like to see what BAM will produce based on
            //previous experiance....
            //inputs			//expected values, not shown to BAM
            SetPattern(patterns[0], 1, 1, 1, 1, -1, -1, -1, -1);
            SetPattern(patterns[1], 1, 1, 1, -1, -1, -1, -1, 1);
            SetPattern(patterns[2], 1, -1, 1, -1, -1, 1, -1, 1);
            SetPattern(patterns[3], -1, 1, -1, 1, 1, -1, 1, -1);

            Console.Out.WriteLine("Input pattern:           BAM output:         Expected output:  ");
            foreach (var p in patterns)
            {
                BAMSystem.SetValuesFromPattern(p);
                BAMSystem.Run();
                foreach (var d in p.Input)
                    Console.Out.Write(d + ",");
                Console.Out.Write("             ");
                for (var i = 0; i < BAMSystem.OutputNodesCount; i++)
                    Console.Out.Write(BAMSystem.OutputNode(i).Value + ",");
                Console.Out.Write("           ");
                foreach (var d in p.Output)
                    Console.Out.Write(d + ",");
                Console.Out.WriteLine("           ");
            }
            Console.In.ReadLine();
        }
    }
}