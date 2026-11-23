% The design of this repository's papers. A paper loads it with \documentclass{sdt-paper} and holds only content, so
% changing the look, or moving to a venue's template, means changing this file and not the paper.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{sdt-paper}[2026/11/23 State-dependent twins paper design]
\LoadClass[11pt]{article}

% Every gap set here is the 10pt text of tables and figures times a power of 1.272, the square root of the golden
% ratio: 0.71, 1.15, 1.46, 1.86, 2.36, 3.00, 3.82, 4.86, 6.18, 7.86, 10, 12.72, 16.18, 20.58, 26.18. Rule weights step
% by the golden ratio: 0.31, 0.5, 0.81. A gap between blocks may stretch or shrink by the step six below it (the gap
% over the golden ratio cubed, about 24%), so LaTeX can balance pages; a gap within a block is fixed. Sizes of the page,
% figures and the footnote rule are golden-ratio proportions. Font sizes and line spacing are LaTeX's own.
% sdt.mplstyle sets figures on the same ladder.
%
% Vertical tokens: 20.58 separates major blocks, 16.18 the next level, 12.72 the level below, 10 sets off an inset
% block (equation, list, footnotes), 7.86 a section heading from its text, 6.18 a smaller heading or caption from its
% content, 4.86 items within a block, 3.00 and 1.86 fine internal spacing. Horizontal tokens: 16.18 is the one indent,
% 10 a gap between columns, 4.86 a label from its text.

% Fonts: Latin Modern, the Computer Modern of LaTeX's default look, with microtype's character protrusion and font
% expansion. inputs.py typesets the figures' text in the same fonts.
\RequirePackage[T1]{fontenc}
\RequirePackage{lmodern}
\RequirePackage[protrusion=true,expansion=true]{microtype}
\frenchspacing

% Page: US letter with a 360pt (5in) text width, LaTeX's article default at 11pt, for about 74 characters a line. The
% text height is the width times the golden ratio, rounded to whole lines, and the top and bottom margins split the
% rest 1 to the golden ratio (89:144). The page number's baseline sits 26.18pt below the text block on every page.
% sdt.mplstyle sizes figures to this text width.
\RequirePackage[letterpaper,textwidth=360pt,textheight=582pt,heightrounded,vmarginratio=89:144,footskip=26.18pt]{geometry}
\clubpenalty=10000
\widowpenalty=10000
\displaywidowpenalty=10000

% Paragraphs: the indent marks a new paragraph, with no space between paragraphs
\setlength\parindent{16.18pt}
\setlength\parskip{0pt}

% Manual skips
\setlength\smallskipamount{3.00pt plus 0.71pt minus 0.71pt}
\setlength\medskipamount{6.18pt plus 1.46pt minus 1.46pt}
\setlength\bigskipamount{12.72pt plus 3.00pt minus 3.00pt}
\setlength\fboxsep{3.00pt}

% Lists: every level indents by the one indent, with 4.86pt from a label to its text. A list is set off like an
% equation (10pt), and items at every level, and a nested list, are 4.86pt apart.
\setlength\leftmargini{16.18pt}
\setlength\leftmarginii{16.18pt}
\setlength\leftmarginiii{16.18pt}
\setlength\leftmarginiv{16.18pt}
\setlength\leftmarginv{16.18pt}
\setlength\leftmarginvi{16.18pt}
\setlength\labelsep{4.86pt}
\setlength\partopsep{0pt}
\def\@listI{\leftmargin\leftmargini \parsep\z@ \partopsep\z@
  \topsep 10\p@ \@plus 2.36\p@ \@minus 2.36\p@ \itemsep 4.86\p@ \@plus 1.15\p@ \@minus 1.15\p@}
\def\@listii{\leftmargin\leftmarginii \labelwidth\leftmarginii \advance\labelwidth-\labelsep \parsep\z@ \partopsep\z@
  \topsep 4.86\p@ \@plus 1.15\p@ \@minus 1.15\p@ \itemsep 4.86\p@ \@plus 1.15\p@ \@minus 1.15\p@}
\def\@listiii{\leftmargin\leftmarginiii \labelwidth\leftmarginiii \advance\labelwidth-\labelsep \parsep\z@ \partopsep\z@
  \topsep 4.86\p@ \@plus 1.15\p@ \@minus 1.15\p@ \itemsep 4.86\p@ \@plus 1.15\p@ \@minus 1.15\p@}

% Displayed equations: set off by 10pt, or none above and 6.18pt below when the line before is short. LaTeX resets
% these and the first list level with every change of font size, so the settings are added to each size the paper uses.
\def\sdt@spacing{%
  \abovedisplayskip 10\p@ \@plus 2.36\p@ \@minus 2.36\p@
  \belowdisplayskip \abovedisplayskip
  \abovedisplayshortskip \z@
  \belowdisplayshortskip 6.18\p@ \@plus 1.46\p@ \@minus 1.46\p@
  \let\@listi\@listI}
\@for\sdt@size:=normalsize,small,footnotesize\do{%
  \expandafter\g@addto@macro\csname\sdt@size\space\endcsname{\sdt@spacing}}
\normalsize

% Footnotes: set off by 10pt, under a 0.5pt rule as long as the text width over the golden ratio squared. A footnote's
% text starts at the one indent, with its mark right-aligned before it.
\setlength{\skip\footins}{10pt plus 2.36pt minus 2.36pt}
\renewcommand\footnoterule{\kern-3.00\p@ \hrule \@height 0.5\p@ \@width 0.381966\columnwidth \kern 2.36\p@}
\renewcommand\@makefntext[1]{\parindent 16.18\p@ \noindent \hb@xt@16.18\p@{\hss\@makefnmark}#1}

% Title block. Its last line is the version line, \sdtversion, which papers/render writes to build/version.tex: the
% release's date, or the date and commit of a draft.
\InputIfFileExists{build/version.tex}{}{\newcommand\sdtversion{Draft}}
\renewcommand\@maketitle{%
  {\centering
    {\LARGE\bfseries\@title\par}
    \vskip 12.72pt
    {\large\@author\par}
    \vskip 6.18pt
    {\small\sdtversion\par}}
  \vskip 20.58pt plus 4.86pt minus 4.86pt}

% Headings
\RequirePackage{titlesec}
\titleformat{\section}{\large\bfseries}{\thesection}{10pt}{}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{10pt}{}
\titlespacing*{\section}{0pt}{16.18pt plus 3.82pt minus 3.82pt}{7.86pt}
\titlespacing*{\subsection}{0pt}{12.72pt plus 3.00pt minus 3.00pt}{6.18pt}

% Math, tables and figures. A table or figure spans the text width, or the text width over the golden ratio with a
% caption of the same width. Its text is \small, as is its caption: above a table, below a figure.
\RequirePackage{amsmath}
\RequirePackage{booktabs}
\RequirePackage{graphicx}
\RequirePackage{pgf}
\RequirePackage{caption}
\DeclareCaptionLabelSeparator{sdt}{.\hspace{4.86pt}}
\captionsetup{font=small,labelfont=bf,labelsep=sdt,skip=6.18pt}
\captionsetup[table]{position=top}
\g@addto@macro\@floatboxreset{\small}   % after LaTeX resets each table and figure to \normalsize
\setlength\tabcolsep{5pt}               % the smallest gap between columns is 10pt; full-width tables spread the rest
\setlength\arraycolsep{5pt}             % the same in math arrays
\setlength\heavyrulewidth{0.81pt}       % top and bottom rules
\setlength\lightrulewidth{0.5pt}        % middle rules, as thin as a figure's axes
\setlength\cmidrulewidth{0.31pt}
\setlength\aboverulesep{1.86pt}
\setlength\belowrulesep{3.00pt}
\setlength\defaultaddspace{4.86pt}      % \addlinespace, between groups of rows
\setlength\jot{3.00pt}                  % between the lines of a multi-line equation
\setlength\floatsep{16.18pt plus 3.82pt minus 3.82pt}       % between two floats
\setlength\textfloatsep{20.58pt plus 4.86pt minus 4.86pt}   % between a float and the text
\setlength\intextsep{20.58pt plus 4.86pt minus 4.86pt}
\setlength\@fpsep{16.18pt plus 2fil}                        % between floats on a page of floats only
\renewcommand\topfraction{0.85}
\renewcommand\textfraction{0.15}

% Citations: author-year, formatted by ACL's style, with each title linked to its DOI. Entries are list items, with a
% hanging indent of the one indent.
\RequirePackage[round]{natbib}
\bibliographystyle{acl_natbib}
\setlength\bibsep{4.86pt plus 1.15pt minus 1.15pt}
\setlength\bibhang{16.18pt}
\renewcommand\bibfont{\small}

% Links: dark blue, with the PDF's title and author taken from \title and \author
\RequirePackage{xcolor}
\PassOptionsToPackage{hyphens}{url}
\definecolor{sdtlink}{RGB}{0,62,126}
\RequirePackage[colorlinks,linkcolor=sdtlink,citecolor=sdtlink,urlcolor=sdtlink,pdfusetitle]{hyperref}
\urlstyle{same}
\RequirePackage[capitalise,noabbrev]{cleveref}
