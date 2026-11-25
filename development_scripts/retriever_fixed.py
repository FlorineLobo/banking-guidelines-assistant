
import re
from collections import defaultdict


class Retriever:

    DOCUMENT_ALIASES = {
        "rbi": [
            "rbi",
            "reserve bank of india"
        ],
        "cbj": [
            "cbj",
            "central bank of jordan",
            "jordan"
        ],
        "moodys": [
            "moody's",
            "moodys",
            "moodys ratings"
        ],
    }

    STOP_WORDS = {
        "what", "is", "the", "of", "a", "an",
        "and", "or", "in", "on", "at", "to",
        "for", "from", "with", "by", "this",
        "that", "these", "those", "which",
        "who", "when", "where", "how", "why",
        "does", "do", "did", "are", "was",
        "were", "be", "been", "being",
        "it", "its", "their", "there",
        "they", "them", "we", "you",
        "your", "i", "me", "my",
        "about", "as", "into", "than",
        "then", "also", "can", "could",
        "would", "should", "will"
    }

    def __init__(self, vector_store, embedding_model):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    # --------------------------------------------------
    # 1. QUESTION AND DOCUMENT IDENTIFICATION
    # --------------------------------------------------

    def _tokenize(self, text):
        """Convert text into lowercase tokens and remove stop words."""

        tokens = set(
            re.findall(
                r"\b[a-zA-Z0-9%./'-]+\b",
                text.lower()
            )
        )

        return tokens - self.STOP_WORDS

    def _is_comparison_question(self, question):
        """Identify questions comparing multiple documents."""

        comparison_terms = [
            "compare",
            "comparison",
            "difference",
            "differences",
            "versus",
            "vs",
            "between",
            "across",
        ]

        question_lower = question.lower()

        return (
            any(
                term in question_lower.split()
                for term in comparison_terms
            )
            or "vs." in question_lower
        )

    def _is_date_question(self, question):
        """Identify questions about dates and regulatory timelines."""

        date_terms = [
            "date",
            "when",
            "implementation",
            "implemented",
            "effective",
            "effectiveness",
            "commencement",
            "come into force",
            "take effect",
            "takes effect",
            "entry into force",
            "commencement date",
        ]

        question_lower = question.lower()

        return any(
            term in question_lower
            for term in date_terms
        )

    def _is_commencement_question(self, question):
        """
        Identify questions asking when a regulation or
        set of directions officially comes into force.
        """

        commencement_terms = [
            "effective date",
            "effective dates",
            "date effective",
            "commencement",
            "come into force",
            "comes into force",
            "entry into force",
            "takes effect",
            "take effect",
            "when does it take effect",
            "when will it apply",
            "when do the directions apply",
        ]

        question_lower = question.lower()

        return any(
            term in question_lower
            for term in commencement_terms
        )

    def _get_document_description(self, document_name):
        """
        Build a document identity from its filename and first page.

        We intentionally do not scan every page. A document may mention
        another regulator in its body without being issued by that regulator.
        """

        first_page_text = " ".join(
            chunk["text"]
            for chunk in self.vector_store.chunks
            if (
                chunk["document_name"] == document_name
                and chunk["page_number"] == 1
            )
        )

        return f"{document_name} {first_page_text}".lower()

    def _identify_relevant_documents(self, question):
        """
        Identify documents explicitly mentioned in a question.

        Uses filenames and first-page identifying information rather than
        incidental references to other institutions throughout a PDF.
        """

        question_lower = question.lower()

        available_documents = sorted({
            chunk["document_name"]
            for chunk in self.vector_store.chunks
        })

        matched_documents = []

        for document_name in available_documents:

            description = self._get_document_description(document_name)
            identifiers = [document_name.lower()]

            for abbreviation, aliases in self.DOCUMENT_ALIASES.items():
                if any(alias in description for alias in aliases):
                    identifiers.extend(aliases)
                    identifiers.append(abbreviation)

            if any(
                identifier in question_lower
                for identifier in identifiers
                if len(identifier.strip()) >= 3
            ):
                matched_documents.append(document_name)

        return matched_documents

    # --------------------------------------------------
    # 2. DATE DETECTION AND RELEVANCE SCORING
    # --------------------------------------------------

    def _contains_date(self, text):
        """Detect common numerical and written date formats."""

        date_pattern = (
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
            r"|\b(?:January|February|March|April|May|June|"
            r"July|August|September|October|November|December)"
            r"\s+\d{1,2},?\s+\d{4}\b"
        )

        return bool(
            re.search(
                date_pattern,
                text,
                re.IGNORECASE,
            )
        )

    def _contains_commencement_language(self, text):
        """
        Detect explicit language indicating when a regulation
        or set of directions comes into force.
        """

        commencement_phrases = [
            "shall come into force",
            "come into force with effect from",
            "come into force on",
            "comes into force",
            "shall take effect",
            "shall be effective from",
            "effective from",
            "with effect from",
            "date of commencement",
            "commencement date",
            "entry into force",
        ]

        text_lower = text.lower()

        return any(
            phrase in text_lower
            for phrase in commencement_phrases
        )

    def _calculate_relevance_boost(self, question, chunk_text):
        """
        Prioritise passages containing regulatory dates.

        Explicit commencement provisions receive a stronger
        boost when the user asks for an effective date.

        These are heuristic ranking adjustments, not probabilities.
        """

        if not self._is_date_question(question):
            return 0.0

        chunk_lower = chunk_text.lower()

        regulatory_date_terms = [
            "implementation",
            "implemented",
            "effective",
            "come into force",
            "commencement",
            "shall apply",
            "shall be applicable",
            "shall be implemented",
            "implementation will be",
            "effective from",
            "with effect from",
            "shall take effect",
            "shall be effective",
            "date of commencement",
        ]

        has_regulatory_language = any(
            term in chunk_lower
            for term in regulatory_date_terms
        )

        has_date = self._contains_date(chunk_text)

        has_commencement_language = (
            self._contains_commencement_language(chunk_text)
        )

        # Explicit commencement provision containing its date.
        # Also prioritise it for questions comparing implementation dates.
        if (
            has_commencement_language
            and has_date
            and (
                self._is_commencement_question(question)
                or "implementation" in question.lower()
            )
        ):
            return 0.35

        # Explicit commencement provision without a date.
        if (
            self._is_commencement_question(question)
            and has_commencement_language
        ):
            return 0.20

        # General regulatory date evidence.
        if has_date and has_regulatory_language:
            return 0.12

        # Regulatory language without a date.
        if has_regulatory_language:
            return 0.08

        # Date without regulatory context.
        if has_date:
            return 0.04

        return 0.0

    # --------------------------------------------------
    # 3. HYBRID RETRIEVAL
    # --------------------------------------------------

    def _search_document(
        self,
        question,
        document_name=None,
        top_k=5,
        return_candidates=False,
    ):
        """
        Perform semantic and keyword retrieval.

        Combines:
        - FAISS semantic search
        - Keyword matching
        - Reciprocal Rank Fusion
        - Regulatory relevance boosts

        If document_name is supplied, semantic candidates
        are filtered after searching the complete FAISS index.
        """

        if self.vector_store.index.ntotal == 0:
            return []

        query_embedding = (
            self.embedding_model.generate_query_embedding(
                question
            )
        )

        candidate_count = min(
            self.vector_store.index.ntotal,
            max(top_k * 10, 50),
        )

        # ----------------------------------------------
        # A. SEMANTIC RETRIEVAL
        # ----------------------------------------------

        # IndexFlatIP performs exact similarity search.
        # Searching the complete index before filtering ensures
        # that a document's relevant chunks are not missed.

        semantic_candidates = self.vector_store.search(
            query_embedding,
            top_k=self.vector_store.index.ntotal,
        )

        # Filter only after retrieving the complete candidate set.
        semantic_results = [
            chunk
            for chunk in semantic_candidates
            if (
                document_name is None
                or chunk["document_name"] == document_name
            )
        ]

        # Keep only the required semantic candidate pool.
        semantic_results = semantic_results[:candidate_count]

        semantic_metadata = {}

        for rank, chunk in enumerate(
            semantic_results,
            start=1,
        ):
            semantic_metadata[chunk["chunk_id"]] = {
                "semantic_similarity": chunk["similarity_score"],
                "semantic_rank": rank,
            }

        # ----------------------------------------------
        # B. KEYWORD RETRIEVAL
        # ----------------------------------------------

        question_tokens = self._tokenize(question)

        keyword_scores = []

        for chunk in self.vector_store.chunks:

            if (
                document_name is not None
                and chunk["document_name"] != document_name
            ):
                continue

            chunk_tokens = self._tokenize(chunk["text"])

            overlap = len(
                question_tokens.intersection(chunk_tokens)
            )

            if overlap > 0:
                keyword_scores.append({
                    "chunk": chunk,
                    "keyword_score": overlap,
                })

        keyword_scores.sort(
            key=lambda item: item["keyword_score"],
            reverse=True,
        )

        keyword_results = keyword_scores[:candidate_count]

        # ----------------------------------------------
        # B2. EXPLICIT REGULATORY DATE CANDIDATE INJECTION
        # ----------------------------------------------

        # Semantic and keyword retrieval can miss an important
        # commencement provision. Include passages that contain
        # both an explicit regulatory commencement phrase and a date.
        question_lower = question.lower()
        date_question = self._is_date_question(question)

        needs_regulatory_start_date = (
            self._is_commencement_question(question)
            or "implementation" in question_lower
        )

        if date_question and needs_regulatory_start_date:
            existing_keyword_ids = {
                item["chunk"]["chunk_id"]
                for item in keyword_results
            }

            for chunk in self.vector_store.chunks:
                if (
                    document_name is not None
                    and chunk["document_name"] != document_name
                ):
                    continue

                if chunk["chunk_id"] in existing_keyword_ids:
                    continue

                if (
                    self._contains_date(chunk["text"])
                    and self._contains_commencement_language(
                        chunk["text"]
                    )
                ):
                    keyword_results.append({
                        "chunk": chunk,
                        "keyword_score": 0,
                    })
                    existing_keyword_ids.add(chunk["chunk_id"])

        keyword_metadata = {}

        for rank, item in enumerate(
            keyword_results,
            start=1,
        ):
            chunk = item["chunk"]

            keyword_metadata[chunk["chunk_id"]] = {
                "keyword_overlap": item["keyword_score"],
                "keyword_rank": rank,
            }

        # ----------------------------------------------
        # C. RECIPROCAL RANK FUSION
        # ----------------------------------------------

        rrf_scores = defaultdict(float)
        chunk_lookup = {}

        for rank, chunk in enumerate(
            semantic_results,
            start=1,
        ):
            chunk_id = chunk["chunk_id"]

            rrf_scores[chunk_id] += 1 / (60 + rank)
            chunk_lookup[chunk_id] = chunk

        for rank, item in enumerate(
            keyword_results,
            start=1,
        ):
            chunk = item["chunk"]
            chunk_id = chunk["chunk_id"]

            rrf_scores[chunk_id] += 1 / (60 + rank)
            chunk_lookup[chunk_id] = chunk

        # ----------------------------------------------
        # D. FINAL HYBRID SCORING
        # ----------------------------------------------

        ranked_results = []

        for chunk_id, rrf_score in rrf_scores.items():

            chunk = chunk_lookup[chunk_id]

            semantic_info = semantic_metadata.get(
                chunk_id,
                {}
            )

            keyword_info = keyword_metadata.get(
                chunk_id,
                {}
            )

            relevance_boost = self._calculate_relevance_boost(
                question,
                chunk["text"],
            )

            hybrid_score = rrf_score + relevance_boost

            ranked_results.append({
                **chunk,

                "semantic_similarity": semantic_info.get(
                    "semantic_similarity"
                ),

                "semantic_rank": semantic_info.get(
                    "semantic_rank"
                ),

                "keyword_overlap": keyword_info.get(
                    "keyword_overlap",
                    0
                ),

                "keyword_rank": keyword_info.get(
                    "keyword_rank"
                ),

                "rrf_score": rrf_score,
                "relevance_boost": relevance_boost,
                "hybrid_score": hybrid_score,
                "score": hybrid_score,
            })

        ranked_results.sort(
            key=lambda item: item["hybrid_score"],
            reverse=True,
        )

        if return_candidates:
            return ranked_results[:candidate_count]

        return ranked_results[:top_k]

    # --------------------------------------------------
    # 4. ADJACENT CHUNK EXPANSION
    # --------------------------------------------------

    def _expand_adjacent_chunks(self, results, question):
        """
        Include neighboring chunks when a regulatory statement
        continues into another chunk.

        Neighbors must belong to the same document and page.
        """

        if not self._is_date_question(question):
            return results

        regulatory_phrases = [
            "come into force",
            "shall come into force",
            "implementation will be",
            "shall apply",
            "shall be implemented",
            "effective from",
            "with effect from",
            "commencement",
            "date of commencement",
            "implementation date",
            "shall take effect",
        ]

        chunks_by_id = {
            chunk["chunk_id"]: chunk
            for chunk in self.vector_store.chunks
        }

        expanded_results = list(results)

        existing_ids = {
            result["chunk_id"]
            for result in expanded_results
        }

        for result in results:

            text = result["text"].lower()

            has_regulatory_phrase = any(
                phrase in text
                for phrase in regulatory_phrases
            )

            if not has_regulatory_phrase:
                continue

            if self._contains_date(text):
                continue

            current_id = result["chunk_id"]

            for neighbour_id in [
                current_id - 1,
                current_id + 1,
            ]:

                neighbour = chunks_by_id.get(neighbour_id)

                if neighbour is None:
                    continue

                if (
                    neighbour["document_name"]
                    != result["document_name"]
                ):
                    continue

                if (
                    neighbour["page_number"]
                    != result["page_number"]
                ):
                    continue

                if neighbour_id in existing_ids:
                    continue

                neighbour_score = (
                    result["hybrid_score"] * 0.95
                )

                if self._contains_date(neighbour["text"]):
                    neighbour_score += 0.02

                expanded_results.append({
                    **neighbour,

                    "semantic_similarity": None,
                    "semantic_rank": None,
                    "keyword_overlap": 0,
                    "keyword_rank": None,
                    "rrf_score": None,
                    "relevance_boost": None,

                    "hybrid_score": neighbour_score,
                    "score": neighbour_score,
                })

                existing_ids.add(neighbour_id)

        expanded_results.sort(
            key=lambda item: item["hybrid_score"],
            reverse=True,
        )

        return expanded_results

    # --------------------------------------------------
    # 5. CROSS-DOCUMENT RESULT BALANCING
    # --------------------------------------------------

    def _balanced_document_results(
        self,
        results_by_document,
        top_k,
    ):
        """
        Distribute retrieved passages across selected documents.

        This prevents one document from dominating a
        cross-document comparison.
        """

        balanced_results = []

        document_names = sorted(
            results_by_document.keys()
        )

        max_results = max(
            (
                len(results)
                for results in results_by_document.values()
            ),
            default=0,
        )

        for rank in range(max_results):

            for document_name in document_names:

                document_results = results_by_document[
                    document_name
                ]

                if rank < len(document_results):

                    balanced_results.append(
                        document_results[rank]
                    )

                    if len(balanced_results) >= top_k:
                        return balanced_results

        return balanced_results

    # --------------------------------------------------
    # 6. MAIN RETRIEVAL METHOD
    # --------------------------------------------------

    def retrieve(
        self,
        question,
        top_k=5,
        document_name=None,
    ):
        """
        Retrieve relevant passages from one or more documents.

        Supports:
        1. Searching a selected document.
        2. Searching a document explicitly mentioned by the user.
        3. Comparing explicitly mentioned documents.
        4. Global search across the entire knowledge base.
        """

        if not question or not question.strip():
            raise ValueError(
                "The question cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        date_question = self._is_date_question(question)

        # Case 1: The user explicitly selected a document.
        if document_name:

            results = self._search_document(
                question,
                document_name=document_name,
                top_k=top_k,
                return_candidates=date_question,
            )

            expanded_results = self._expand_adjacent_chunks(
                results,
                question,
            )

            return expanded_results[:top_k]

        available_documents = sorted({
            chunk["document_name"]
            for chunk in self.vector_store.chunks
        })

        if not available_documents:
            return []

        # Identify documents explicitly mentioned in the question.
        relevant_documents = (
            self._identify_relevant_documents(question)
        )

        # Case 2: Compare explicitly mentioned documents.
        if self._is_comparison_question(question):

            if len(relevant_documents) < 2:
                relevant_documents = available_documents

            results_by_document = {}

            for name in relevant_documents:

                results = self._search_document(
                    question,
                    document_name=name,
                    top_k=top_k,
                    return_candidates=date_question,
                )

                expanded_results = self._expand_adjacent_chunks(
                    results,
                    question,
                )

                expanded_results.sort(
                    key=lambda item: item["hybrid_score"],
                    reverse=True,
                )

                results_by_document[name] = expanded_results

            return self._balanced_document_results(
                results_by_document,
                top_k,
            )

        # Case 3: A single document is explicitly mentioned.
        # Restrict retrieval to that document.
        if len(relevant_documents) == 1:

            selected_document = relevant_documents[0]

            results = self._search_document(
                question,
                document_name=selected_document,
                top_k=top_k,
                return_candidates=date_question,
            )

            expanded_results = self._expand_adjacent_chunks(
                results,
                question,
            )

            return expanded_results[:top_k]

        # Case 4: No specific document mentioned.
        # Perform one GLOBAL hybrid search.
        all_results = self._search_document(
            question,
            document_name=None,
            top_k=top_k,
            return_candidates=date_question,
        )

        expanded_results = self._expand_adjacent_chunks(
            all_results,
            question,
        )

        return expanded_results[:top_k]
