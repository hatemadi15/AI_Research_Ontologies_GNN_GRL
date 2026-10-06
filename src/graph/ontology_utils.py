"""
ontology_utils.py - One loader for the Materials Mechanics Ontology (MMO)

Replaces the per-module copies of the rdflib parsing code. For every named
owl:Class it collects:
  - primary label (rdfs:label) and all alternative labels (SKOS / MMO alt/pref)
  - skos:definition (the ontology has 420 class definitions; it has almost no
    rdfs:comment, which is what the old structural expansion looked for)
  - direct named superclasses and the transitive ancestor closure
plus the existential restrictions (class-level "gold" relations) and the
object-property hierarchy.

Only rdflib is needed, so tests and CI can use it without ML dependencies.
"""

import os
from functools import lru_cache

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DEFAULT_TTL_PATH = os.path.join(
    PROJECT_ROOT, "data", "raw", "dataset", "ontologies", "ontology.ttl"
)

MMO_NS = "https://w3id.org/pmd/materials-mechanics-ontology/"
SKOS_NS = "http://www.w3.org/2004/02/skos/core#"
PMD_NS = "https://w3id.org/pmd/co/"
PROV_NS = "http://www.w3.org/ns/prov#"

# Every MMO class sits below this single root, so it carries no information
# and is excluded from hierarchical credit.
ROOT_LOCAL_NAME = "UseCaseMaterialsMechanics"


def local_name(uri):
    """Fragment after the last '#' or '/' of a URI."""
    return str(uri).split('#')[-1].split('/')[-1]


class OntologyIndex:
    """Lookup structure over the classes and axioms of one ontology file."""

    def __init__(self, classes, parents, restrictions, properties):
        # uri -> {'local', 'label', 'labels', 'definition'}
        self.classes = classes
        # uri -> set of direct named superclass URIs (classes in this file only)
        self.parents = parents
        # list of dicts: subject, property, object, inverse (URIs as str)
        self.restrictions = restrictions
        # property uri -> {'label', 'parents', 'domain', 'range'}
        self.properties = properties

        self.uris = sorted(classes)
        root = MMO_NS + ROOT_LOCAL_NAME
        self.root = root if root in classes else None

        self.local_to_uri = {info['local']: uri for uri, info in classes.items()}
        self.label_to_uri = {}
        for uri in self.uris:
            for label in self.classes[uri]['labels']:
                self.label_to_uri.setdefault(label.lower(), uri)
        self._ancestors = {}

    # ---- class lookups -------------------------------------------------
    def primary_label(self, uri):
        return self.classes[uri]['label']

    def labels(self, uri):
        return list(self.classes[uri]['labels'])

    def definition(self, uri):
        return self.classes[uri]['definition']

    def resolve(self, name):
        """Map a label, local name or CamelCase type name to a class URI."""
        if name is None:
            return None
        name = str(name).strip()
        if name in self.classes:
            return name
        if name in self.local_to_uri:
            return self.local_to_uri[name]
        return self.label_to_uri.get(name.lower())

    # ---- hierarchy -----------------------------------------------------
    def ancestors(self, uri, include_self=False, include_root=False):
        """Transitive named superclasses of `uri` within this ontology."""
        if uri not in self._ancestors:
            seen = set()
            stack = list(self.parents.get(uri, ()))
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                stack.extend(self.parents.get(cur, ()))
            seen.discard(uri)
            self._ancestors[uri] = frozenset(seen)
        result = set(self._ancestors[uri])
        if include_self:
            result.add(uri)
        if not include_root and self.root:
            result.discard(self.root)
        return result

    def is_subclass(self, child, parent):
        """True if `parent` is a strict ancestor of `child`."""
        return parent in self.ancestors(child, include_root=True)

    def subclass_closure(self):
        """Set of (child, ancestor) pairs of the subClassOf closure (no root)."""
        return {(c, a) for c in self.uris for a in self.ancestors(c)}

    def reduced_edges(self, active):
        """Transitive reduction of the subClassOf order restricted to `active`.

        Returns (child, parent) pairs where parent is an ancestor of child and
        no other active class lies strictly between them. This is the gold
        standard for class-level taxonomy recall over the classes that actually
        occur in the corpus.
        """
        active = set(active) - ({self.root} if self.root else set())
        edges = set()
        for child in active:
            anc = self.ancestors(child) & active
            for parent in anc:
                between = any(parent in self.ancestors(mid)
                              for mid in anc if mid != parent)
                if not between:
                    edges.add((child, parent))
        return edges

    # ---- properties ----------------------------------------------------
    def property_ancestors(self, prop_uri, include_self=True):
        """Transitive rdfs:subPropertyOf closure of a property URI."""
        seen = set()
        stack = [prop_uri]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(self.properties.get(cur, {}).get('parents', ()))
        if not include_self:
            seen.discard(prop_uri)
        return seen

    def restriction_triples(self):
        """Class-level gold relations as (subject, property, object) triples.

        An existential restriction `C subClassOf (P some V)` yields (C, P, V);
        a restriction on `inverse(P)` yields (V, P, C).
        """
        triples = set()
        for r in self.restrictions:
            if r['inverse']:
                triples.add((r['object'], r['property'], r['subject']))
            else:
                triples.add((r['subject'], r['property'], r['object']))
        return triples


def _collect_labels(g, cls, label_props):
    labels = []
    for prop in label_props:
        for label in sorted(str(lit).strip() for lit in g.objects(cls, prop)):
            if label and not label.startswith('http') and label not in labels:
                labels.append(label)
    return labels


@lru_cache(maxsize=4)
def load_ontology(ttl_path=DEFAULT_TTL_PATH):
    """Parse an ontology TTL file into an OntologyIndex (cached per path)."""
    from rdflib import BNode, Graph, Namespace, OWL, RDF, RDFS, URIRef

    if not os.path.exists(ttl_path):
        raise FileNotFoundError(f"Ontology not found: {ttl_path}")

    g = Graph()
    g.parse(ttl_path, format='turtle')
    MMO = Namespace(MMO_NS)
    SKOS = Namespace(SKOS_NS)
    label_props = [RDFS.label, SKOS.prefLabel, MMO.prefLabel,
                   SKOS.altLabel, MMO.altLabel]

    classes = {}
    for cls in g.subjects(RDF.type, OWL.Class):
        if not isinstance(cls, URIRef):
            continue
        uri = str(cls)
        if uri.startswith('http://www.w3.org/'):
            continue
        labels = _collect_labels(g, cls, label_props)
        if not labels:
            labels = [local_name(uri)]
        definitions = sorted(str(d).strip() for d in g.objects(cls, SKOS.definition))
        classes[uri] = {
            'local': local_name(uri),
            'label': labels[0],
            'labels': labels,
            'definition': definitions[0] if definitions else '',
        }

    parents = {}
    for s, o in g.subject_objects(RDFS.subClassOf):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            if str(s) in classes and str(o) in classes:
                parents.setdefault(str(s), set()).add(str(o))

    restrictions = []
    for r in g.subjects(RDF.type, OWL.Restriction):
        on_prop = next(iter(g.objects(r, OWL.onProperty)), None)
        if on_prop is None:
            continue
        inverse = False
        if isinstance(on_prop, BNode):
            inv = next(iter(g.objects(on_prop, OWL.inverseOf)), None)
            if not isinstance(inv, URIRef):
                continue
            on_prop, inverse = inv, True
        values = list(g.objects(r, OWL.someValuesFrom)) + \
            list(g.objects(r, OWL.allValuesFrom))
        for subj in g.subjects(RDFS.subClassOf, r):
            if str(subj) not in classes:
                continue  # restrictions on imported (PMDco/PROV) classes
            for val in values:
                if isinstance(val, URIRef):
                    restrictions.append({
                        'subject': str(subj),
                        'property': str(on_prop),
                        'object': str(val),
                        'inverse': inverse,
                    })

    properties = {}
    for prop in set(g.subjects(RDF.type, OWL.ObjectProperty)) | \
            set(s for s, _ in g.subject_objects(RDFS.subPropertyOf)):
        if not isinstance(prop, URIRef):
            continue
        labels = _collect_labels(g, prop, [RDFS.label])
        properties[str(prop)] = {
            'label': labels[0] if labels else local_name(prop),
            'parents': {str(p) for p in g.objects(prop, RDFS.subPropertyOf)
                        if isinstance(p, URIRef)},
            'domain': {str(d) for d in g.objects(prop, RDFS.domain)
                       if isinstance(d, URIRef)},
            'range': {str(r) for r in g.objects(prop, RDFS.range)
                      if isinstance(r, URIRef)},
        }

    return OntologyIndex(classes, parents, restrictions, properties)
