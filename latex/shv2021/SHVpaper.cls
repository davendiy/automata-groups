\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{SHVpaper}
\newif\ifTAAC@russian
\newif\ifTAAC@ukrainian
\newif\ifTAAC@english
\newif\ifTAAC@changefont
\TAAC@russianfalse
\TAAC@ukrainiantrue
\TAAC@englishfalse
\TAAC@changefontfalse
\DeclareOption{russian}{\TAAC@russiantrue \TAAC@ukrainianfalse \TAAC@englishfalse}
\DeclareOption{ukrainian}{\TAAC@russianfalse \TAAC@ukrainiantrue \TAAC@englishfalse}
\DeclareOption{english}{\TAAC@russianfalse \TAAC@ukrainianfalse \TAAC@englishtrue}
\ProcessOptions

\LoadClass[a5paper, 10pt]{book}
\RequirePackage[cp1251]{inputenc}
\RequirePackage[left=1cm,right=1cm,top=1cm,bottom=1.5cm]{geometry}
\RequirePackage[font=small,format=plain,labelfont=bf,up, skip=2pt, labelsep=period]{caption}
\RequirePackage[stable]{footmisc}
\RequirePackage{graphicx}
\RequirePackage[english,russian,ukrainian]{babel}
\RequirePackage{cmap}
\RequirePackage{amsmath}

\ifTAAC@russian
%\RequirePackage[russian]{babel}
\fi

\ifTAAC@ukrainian
%\RequirePackage[ukrainian]{babel}
\fi

\ifTAAC@english
%\RequirePackage[english]{babel}
\fi

\addto\captionsukrainian{
    \renewcommand\tablename{Таблиця}
    \renewcommand\refname{Список літератури}
}

\pagestyle{empty}

\parindent=.8cm

%\renewcommand{\baselinestretch}{1.0}

%\renewcommand\large{\@setfontsize\large\@xiipt{12}}

\setcounter{footnote}{0}

\renewcommand\thesection{}
\renewcommand\thesubsection{}
\renewcommand\section{%
   \pagebreak%
   \setcounter{table}{0}%
   \setcounter{figure}{0}%
   \setcounter{equation}{0}%
   \setcounter{footnote}{0}%
   \@startsection {section}{1}{\z@}%
   {-3.5ex \@plus -1ex \@minus -.2ex}%
   {2.3ex \@plus.2ex}%
   {\center\normalfont%
   \ifTAAC@changefont%
      \usefont{T2A}{oarial}{b}{n}\fontsize{12}{12}\selectfont
   \else
     \Large\sffamily\bfseries
   \fi}}

\emergencystretch = 25pt
\tolerance=400

\newcommand\subsectiontemp{ \@startsection{subsection}{2}{\z@}%
   {0.5ex}%
   {.1ex}%
   {
   \hrule
   \nopagebreak%
   \hspace{-0.5cm}\strut\rule{0pt}{12pt}
   \ifTAAC@changefont%
      \normalfont \usefont{T2A}{oarial}{b}{n} \fontsize{10}{12} \selectfont
   \else
      \normalfont\sffamily\bfseries%
   \fi
   }}

\renewcommand\subsection[1]{
  \vspace*{2ex \@plus -0ex \@minus -.0ex}%
  \subsectiontemp{#1}\nopagebreak%
  \hrule
  \nopagebreak%
  \vspace*{1.5ex \@plus .2ex} }

\newcommand\authors[1]{\par
    {\centering{\normalfont\bfseries#1}\par\vspace*{0.05in}}
}

\newcommand\abstract[1]{\par
    {{\normalfont\itshape#1}\par\vspace*{0.05in}}
}
\renewenvironment{thebibliography}[1]
     {\subsection{\refname}%
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}

\renewcommand\author[3]{\par{\bfseries#1} --- #2; \mbox{E-mail:~\underline{\itshape #3}}}

\renewcommand{\@listI}{%
\leftmargin=1.2cm
\rightmargin=0pt
\labelsep=5pt
\labelwidth=20pt
\itemindent=0pt
\listparindent=0pt
\topsep=0pt plus 2pt minus 0pt
\partopsep=0pt plus 1pt minus 0pt
\parsep=0pt plus 1pt
\itemsep=\parsep}
\renewcommand{\@listii}{%
  \@listI%
  \leftmargin=.4cm%
}
\renewcommand{\@listiii}{\@listii}
\renewcommand{\@listiv}{\@listii}
\renewcommand{\@listv}{\@listii}
\renewcommand{\@listvi}{\@listii}

