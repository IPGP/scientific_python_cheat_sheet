#!/usr/bin/env python

"""
A script to generate a file named index.html from sheet.md and templace.html
"""

from bs4 import BeautifulSoup
import markdown
import subprocess


top = """
<!DOCTYPE html>
<html lang="en-us">
  <head>
    <meta charset="UTF-8">
    <title>Scientific python cheat sheet by IPGP</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.5">
    <link rel="stylesheet" type="text/css" href="stylesheets/normalize.css" media="screen">
    <link href='https://fonts.googleapis.com/css?family=Open+Sans:400,700' rel='stylesheet' type='text/css'>
    <link rel="stylesheet" type="text/css" href="stylesheets/stylesheet.css" media="screen">
    <link rel="stylesheet" type="text/css" href="stylesheets/github-light.css" media="screen">
    <link rel="stylesheet" type="text/css" href="github.css" media="screen">
  <style type="text/css">
div.sourceCode { overflow-x: auto; }
table.sourceCode, tr.sourceCode, td.lineNumbers, td.sourceCode {
  margin: 0; padding: 0; vertical-align: baseline; border: none; }
table.sourceCode { width: 100%; line-height: 100%; }
td.lineNumbers { text-align: right; padding-right: 4px; padding-left: 4px; color: #aaaaaa; border-right: 1px solid #aaaaaa; }
td.sourceCode { padding-left: 5px; }
code.sourceCode span.hll { background-color: #ffffcc }
code > span.c { color: #999988; font-style: italic } /* Comment */
code > span.err { color: #a61717; background-color: #e3d2d2 } /* Error */
code > span.k { color: #000000; font-weight: bold } /* Keyword */
code > span.o { color: #000000; font-weight: bold } /* Operator */
code > span.cm { color: #999988; font-style: italic } /* Comment.Multiline */
code > span.cp { color: #999999; font-weight: bold; font-style: italic } /* Comment.Preproc */
code > span.c1 { color: #999988; font-style: italic } /* Comment.Single */
code > span.cs { color: #999999; font-weight: bold; font-style: italic } /* Comment.Special */
code > span.gd { color: #000000; background-color: #ffdddd } /* Generic.Deleted */
code > span.ge { color: #000000; font-style: italic } /* Generic.Emph */
code > span.gr { color: #aa0000 } /* Generic.Error */
code > span.gh { color: #999999 } /* Generic.Heading */
code > span.gi { color: #000000; background-color: #ddffdd } /* Generic.Inserted */
code > span.go { color: #888888 } /* Generic.Output */
code > span.gp { color: #555555 } /* Generic.Prompt */
code > span.gs { font-weight: bold } /* Generic.Strong */
code > span.gu { color: #aaaaaa } /* Generic.Subheading */
code > span.gt { color: #aa0000 } /* Generic.Traceback */
code > span.kc { color: #000000; font-weight: bold } /* Keyword.Constant */
code > span.kd { color: #000000; font-weight: bold } /* Keyword.Declaration */
code > span.kn { color: #000000; font-weight: bold } /* Keyword.Namespace */
code > span.kp { color: #000000; font-weight: bold } /* Keyword.Pseudo */
code > span.kr { color: #000000; font-weight: bold } /* Keyword.Reserved */
code > span.kt { color: #445588; font-weight: bold } /* Keyword.Type */
code > span.m { color: #009999 } /* Literal.Number */
code > span.s { color: #d01040 } /* Literal.String */
code > span.na { color: #008080 } /* Name.Attribute */
code > span.nb { color: #0086B3 } /* Name.Builtin */
code > span.nc { color: #445588; font-weight: bold } /* Name.Class */
code > span.no { color: #008080 } /* Name.Constant */
code > span.nd { color: #3c5d5d; font-weight: bold } /* Name.Decorator */
code > span.ni { color: #800080 } /* Name.Entity */
code > span.ne { color: #990000; font-weight: bold } /* Name.Exception */
code > span.nf { color: #990000; font-weight: bold } /* Name.Function */
code > span.nl { color: #990000; font-weight: bold } /* Name.Label */
code > span.nn { color: #555555 } /* Name.Namespace */
code > span.nt { color: #000080 } /* Name.Tag */
code > span.nv { color: #008080 } /* Name.Variable */
code > span.ow { color: #000000; font-weight: bold } /* Operator.Word */
code > span.w { color: #bbbbbb } /* Text.Whitespace */
code > span.mf { color: #009999 } /* Literal.Number.Float */
code > span.mh { color: #009999 } /* Literal.Number.Hex */
code > span.mi { color: #009999 } /* Literal.Number.Integer */
code > span.mo { color: #009999 } /* Literal.Number.Oct */
code > span.sb { color: #d01040 } /* Literal.String.Backtick */
code > span.sc { color: #d01040 } /* Literal.String.Char */
code > span.sd { color: #d01040 } /* Literal.String.Doc */
code > span.s2 { color: #d01040 } /* Literal.String.Double */
code > span.se { color: #d01040 } /* Literal.String.Escape */
code > span.sh { color: #d01040 } /* Literal.String.Heredoc */
code > span.si { color: #d01040 } /* Literal.String.Interpol */
code > span.sx { color: #d01040 } /* Literal.String.Other */
code > span.sr { color: #009926 } /* Literal.String.Regex */
code > span.s1 { color: #d01040 } /* Literal.String.Single */
code > span.ss { color: #990073 } /* Literal.String.Symbol */
code > span.bp { color: #999999 } /* Name.Builtin.Pseudo */
code > span.vc { color: #008080 } /* Name.Variable.Class */
code > span.vg { color: #008080 } /* Name.Variable.Global */
code > span.vi { color: #008080 } /* Name.Variable.Instance */
code > span.il { color: #009999 } /* Literal.Number.Integer.Long */
  </style>
  </head>
  <body>
    <section class="main-content">
"""


bottom = """
      <footer class="site-footer">
        <span class="site-footer-owner"><a href="https://github.com/IPGP/scientific_python_cheat_sheet">Scientific python cheat sheet</a> is maintained by <a href="https://github.com/IPGP">IPGP</a>.</span>

        <span class="site-footer-credits">This page was generated by <a href="https://pages.github.com">GitHub Pages</a> using the <a href="https://github.com/jasonlong/cayman-theme">Cayman theme</a> by <a href="https://twitter.com/jasonlong">Jason Long</a>.</span>
      </footer>

    </section>
  </body>
</html>
"""

input_file = "sheet.md"
output_file = "sheet.html"

try:
    cmd = "pandoc {} -s -o {}".format(input_file,
                                      output_file)
    print cmd
    subprocess.call(cmd, shell=True)
    soup_sheet = BeautifulSoup(open(output_file), "html.parser")
    list_html = map(str, list(soup_sheet.body.children))
    sheet_html = ''.join(list_html)
    print "went the pandoc way"
except:
    print "pandoc failed, using shitty markdown"
    with open(input_file, "r") as f:
        text = f.read()
    sheet_html = markdown.markdown(text)

sheet_html_lines = sheet_html.split("\n")
f = open("index.html", "w")
flag_first_h2 = False
f.write(top+"\n")
for line in sheet_html_lines:
    if "markdown-toc" in line:
        continue

    if "h2" in line:
        if flag_first_h2:
            f.write("</div>\n<div class=group>\n")
            f.write(line+"\n")
        else:
            f.write("<div class=group>\n")
            f.write(line+"\n")
            flag_first_h2 = True
    else:
        f.write(line+"\n")

f.write("</div>\n")
f.write(bottom+"\n")
f.close()
