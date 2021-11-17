$$
\begin{aligned}
\begin{align*}
L \xmapsto{} G(L, \varnothing) \xmapsto{\text{PubMed}} \text{\ documents} \xmapsto{\text{USEv4}}\vec{X} \xmapsto{\text{UMAP}}\vec{U} \xmapsto{\text{HDBSCAN}} \vec{W} \xmapsto{pool} \vec{Z} \xmapsto{\text{similarity}} H(L,e^{(1)}) \xmapsto{\text{metapath2vec}}  G(L,e^{(2)}) \xmapsto{\text{link clustering}} \cnums

\\ \\
\end{align*}
\end{aligned}

\\ \\

\begin{aligned}
\begin{align*}

\hline \\


L &= L_{\text{tasks}} \cup L_{\text{constructs}} & \text{labels}\\

\text{documents} &= \big\{ (\text{abstract},{l_i})\ \big| \ l_i \in L \big\} \\

|\vec{X}| &\in \R^{\text{|documents|} \times 768} & \text{document embeddings}\\

|\vec{U}| &\in \R^{\text{|documents|}\times 5} & \text{reduced document embeddings}\\

\big|{e^{(1)}}\big| &\in \R^{\text{|L|} \times \text{|L|} \times \text{|topics|}} & \text{affinity matrix} \\

\big|{e^{(2)}}\big| &\in \R^{\text{|L|} \times \text{|L|} \times 256} & \text{affinity matrix} \\

\cnums &= \{ (c_i,l_j) \ \big| \ c_i \in \N, l_j \in L \big\} & \text{overlapping communities}\\

\end{align*}
\end{aligned}
