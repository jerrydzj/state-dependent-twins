% The design of this repository's papers. A paper loads it with \documentclass{sdt-paper} and holds only content, so
% changing the look, or moving to a venue's template, means changing this file and not the paper.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{sdt-paper}[2026/11/23 State-dependent twins paper design]
\LoadClass[11pt]{article}

% Fonts: Latin Modern, the Computer Modern of LaTeX's default look, with microtype's character protrusion and font
% expansion. inputs.py typesets the figures' text in the same fonts.
\RequirePackage[T1]{fontenc}
\RequirePackage{lmodern}
\RequirePackage[protrusion=true,expansion=true]{microtype}
\frenchspacing

% Page: US letter with a 6in by 9in text block. inputs.py sizes the figures to this text width.
\RequirePackage[letterpaper,textwidth=6in,textheight=9in,centering]{geometry}
\clubpenalty=10000
\widowpenalty=10000
\displaywidowpenalty=10000

% Title block. Its last line is the version line, \sdtversion, which papers/render writes to build/version.tex: the
% release's date, or the date and commit of a draft.
\InputIfFileExists{build/version.tex}{}{\newcommand\sdtversion{Draft}}
\renewcommand\@maketitle{%
  \begin{center}
    {\LARGE\bfseries\@title\par}
    \vskip 1.2em
    {\large\@author\par}
    \vskip 0.6em
    {\small\sdtversion\par}
  \end{center}
  \vskip 1.5em}

% Headings
\RequirePackage{titlesec}
\titleformat{\section}{\large\bfseries}{\thesection}{0.8em}{}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{0.8em}{}
\titlespacing*{\section}{0pt}{2.6ex plus 1ex minus 0.2ex}{1.2ex plus 0.2ex}
\titlespacing*{\subsection}{0pt}{2ex plus 1ex minus 0.2ex}{0.8ex plus 0.2ex}

% Math, tables and figures. Table captions go above the table, figure captions below the figure.
\RequirePackage{amsmath}
\RequirePackage{booktabs}
\RequirePackage{graphicx}
\RequirePackage{pgf}
\RequirePackage[font=small,labelfont=bf,labelsep=period,skip=6pt]{caption}
\captionsetup[table]{position=top}
\AddToHook{env/table/begin}{\small}
\renewcommand\topfraction{0.85}
\renewcommand\textfraction{0.15}

% Citations: author-year, formatted by ACL's style, with each title linked to its DOI
\RequirePackage[round]{natbib}
\bibliographystyle{acl_natbib}
\setlength\bibsep{0.6ex}
\renewcommand\bibfont{\small}

% Links: dark blue, with the PDF's title and author taken from \title and \author
\RequirePackage{xcolor}
\PassOptionsToPackage{hyphens}{url}
\definecolor{sdtlink}{RGB}{0,62,126}
\RequirePackage[colorlinks,linkcolor=sdtlink,citecolor=sdtlink,urlcolor=sdtlink,pdfusetitle]{hyperref}
\urlstyle{same}
\RequirePackage[capitalise,noabbrev]{cleveref}
