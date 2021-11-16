$$
\begin{aligned}
\begin{align*}
L &\xmapsto{} G(L, \varnothing) \xmapsto{\text{PubMed}} \text{documents} \\ \\

\text{\ documents} &\xmapsto{\text{USEv4}}\vec{X} \xmapsto{\text{UMAP}}\vec{U} \xmapsto{\text{HDBSCAN}}\vec{\cnums},\vec{W} \xmapsto{pool} \vec{Z} \\ \\

\vec{Z} &\xmapsto{\text{outer product}} \vec{H} \xmapsto{\text{metapath2vec}} G(L, e) \\ \\

G(L,e) &\xmapsto{\text{spectral clustering}} \text{overlapping communities}
\\ \\
\hline \\


L &= L_{\text{tasks}} \cup L_{\text{constructs}} & \text{labels}\\

\text{documents} &= \bigg\{ (\text{abstract},{l_i})\ \bigg| \ l_i \in L \bigg\} \\

|\vec{X}| &\in \R^{\text{|documents|} \times 768} & \text{document embeddings}\\

|\vec{U}| &\in \R^{\text{|documents|}\times 5} & \text{reduced document embeddings}\\

|\vec{\cnums}| &\in \N^{\text{|documents|}} & \text{topics}\\

|\vec{H}| &\in \R^{\text{|L|} \times \text{|topics|}} & \text{topic embeddings} \\

|\vec{Z}| &\in \R^{\text{|L|} \times \text{|L|}} & \text{affinity matrix}


\end{align*}
\end{aligned}
