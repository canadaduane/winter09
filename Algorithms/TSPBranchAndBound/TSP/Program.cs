using System;
using System.Collections.Generic;
using System.Windows.Forms;

namespace TSP
{
    static class Program
    {
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
       {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(MainForm = new Form1());
        }
        public static Form1 MainForm;

    }
}