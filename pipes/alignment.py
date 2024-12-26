import pandas as pd
from xml.etree.ElementTree import Element, SubElement, tostring, fromstring, ParseError, parse, ElementTree
import re
import os
from bertalign import Bertalign


class AlignmentPipeline:
    def __init__(self, texts_path, xml_output_dir):
        self.texts_path = texts_path
        self.xml_output_dir = xml_output_dir
        os.makedirs(xml_output_dir, exist_ok=True)
        self.texts_data = None
        self.aligned_texts = []  # Updated to hold tuples of (src_id, tgt_id, pair_name)
        self.failed_pairs = []

    def load_data(self):
        """Load and preprocess input data."""
        self.texts_data = pd.read_excel(self.texts_path, dtype={'texts.id': str})
        self.texts_data.rename(columns={
            'texts.id': 'id',
            'texts.event_id': 'event_id',
            'texts.lang': 'lang',
            'texts.source_target': 'source_target',
            'texts.spoken_written': 'spoken_written',
            'texts.sentence_split_text': 'sentence_split_text'
        }, inplace=True)

        # Process sentence_split_text into processed_text
        self.texts_data['processed_text'] = self.texts_data['sentence_split_text'].apply(self._process_xml_sentences)

    @staticmethod
    def _process_xml_sentences(xml_str):
        """Extract text content from XML sentences."""
        if pd.isnull(xml_str):
            return ""
        try:
            root = fromstring(xml_str)
            sentences = [s.text for s in root.findall('.//s') if s.text is not None]
            return '\n'.join(sentences)
        except ParseError:
            return ""

    def align_texts(self):
        """Align texts and save results in memory."""
        grouped = self.texts_data.groupby('event_id')

        print("Aligning files (this will take some time)...")

        for event_id, group in grouped:
            combinations = group.groupby(['lang', 'source_target', 'spoken_written'])

            for (lang1, src_target1, spoken_written1), group1 in combinations:
                for (lang2, src_target2, spoken_written2), group2 in combinations:
                    if (lang1, src_target1, spoken_written1) < (lang2, src_target2, spoken_written2):
                        # Generate pair name in the new lowercase template
                        pair_name = (
                            f"EPTIC.{lang1.lower()}_{spoken_written1.lower()}_{src_target1.lower()}."
                            f"{lang2.lower()}_{spoken_written2.lower()}_{src_target2.lower()}"
                        )
                        alignment_results = []

                        # Align all sentences in the pair
                        for row1 in group1.itertuples(index=False):
                            for row2 in group2.itertuples(index=False):
                                alignment_result = self._align_sents(row1.id, row2.id, lang1, lang2)
                                if alignment_result:
                                    alignment_results.extend(alignment_result)
                                    self.aligned_texts.append((row1.id, row2.id, pair_name))  # Include pair name
                                else:
                                    self.failed_pairs.append((row1.id, row2.id, pair_name))

                        # Generate XML for the pair
                        if alignment_results:
                            self._generate_xml(pair_name, alignment_results)

    def _align_sents(self, src_id, tgt_id, src_lang, tgt_lang):
        """Align sentences using Bertalign."""
        src_text = self.texts_data.loc[self.texts_data['id'] == src_id, 'processed_text'].values[0]
        tgt_text = self.texts_data.loc[self.texts_data['id'] == tgt_id, 'processed_text'].values[0]

        if not src_text or not tgt_text:
            return None

        aligner = Bertalign(src_text, tgt_text, src_lang=src_lang, tgt_lang=tgt_lang)
        aligner.align_sents()

        alignments, _ = aligner.get_result()
        results = []

        for left_side, right_side in alignments:
            src_targets = ' '.join([f"{src_id}:{idx}" for idx in left_side])
            tgt_targets = ' '.join([f"{tgt_id}:{idx}" for idx in right_side])
            results.append({
                "xtargets": f"{src_targets};{tgt_targets}",
                "type": f"{len(left_side)}-{len(right_side)}",
                "status": "auto"
            })

        return results

    def _generate_xml(self, pair_name, alignment_results):
        """Generate or append XML for all alignments of a pair."""
        output_path = os.path.join(self.xml_output_dir, f"{pair_name}.xml")

        # Limit the results to only those relevant for the given pair_name
        alignment_results = [result for result in alignment_results if "643:0" in result["xtargets"] or "641:" in result["xtargets"]]

        # Create a new root element
        root = Element('linkGrp', attrib={'toDoc': 'placeholder_toDoc.xml', 'fromDoc': 'placeholder_fromDoc.xml'})

        # Append only the necessary alignment results
        for result in alignment_results:
            SubElement(root, 'link', attrib={
                'type': result['type'],
                'xtargets': result['xtargets'],
                'status': result['status']
            })

        # Prettify, clean, and save the XML
        xml_content = self._prettify_and_refine_xml(tostring(root, encoding='unicode'))
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)

    @staticmethod
    def _prettify_and_refine_xml(xml_string):
        """Prettify and clean the XML output."""
        # Rearrange link attributes
        xml_string = AlignmentPipeline._rearrange_link_attributes(xml_string)
        # Remove extra spaces and line breaks
        xml_string = re.sub(r'>\s*<', '>\n<', xml_string).strip()
        # Ensure all quotes are single quotes
        xml_string = xml_string.replace('"', "'")
        return f"<?xml version='1.0' encoding='utf-8'?>\n{xml_string}"

    @staticmethod
    def _rearrange_link_attributes(xml_string):
        """Rearrange attributes in <link> tags."""
        tree = ElementTree(fromstring(xml_string))
        root = tree.getroot()
        for link in root.findall('.//link'):
            attributes = link.attrib
            # Order attributes as 'type', 'xtargets', and 'status'
            ordered_attributes = {k: attributes.pop(k) for k in ['type', 'xtargets', 'status'] if k in attributes}
            link.attrib.clear()
            link.attrib.update(ordered_attributes)
        return tostring(root, encoding='unicode').replace(' />', '/>')

    def save_alignment_file(self):
        """Save alignments to an Excel file."""
        rows = []
        for src_id, tgt_id, pair_name in self.aligned_texts:
            file_path = os.path.join(self.xml_output_dir, f"{pair_name}.xml")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                rows.append({'t1_id': src_id, 't2_id': tgt_id, 'alignment_file': content})

        output_file = os.path.join(self.xml_output_dir, "alignments.xlsx")
        pd.DataFrame(rows).to_excel(output_file, index=False)
        print(f"Alignment details saved to {output_file}")

    def run(self):
        """Execute the pipeline."""
        self.load_data()
        self.align_texts()
        if self.failed_pairs:
            print("Warning: The following pairs failed alignment:", self.failed_pairs)
        self.save_alignment_file()


if __name__ == "__main__":
    pipeline = AlignmentPipeline(
        texts_path='/home/afedotova/bertalign/database/texts.xlsx',
        xml_output_dir='/home/afedotova/bertalign/alignment_repo/output2'
    )
    pipeline.run()
