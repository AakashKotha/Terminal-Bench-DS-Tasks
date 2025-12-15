import os
import json
import time
import glob
from heapq import merge

class MemTable:
    def __init__(self):
        self.data = {}
        # Estimated size in bytes
        self.size = 0 

    def put(self, key, value):
        # We store (value, timestamp, is_deleted)
        # For a put, is_deleted is False
        ts = time.time()
        self.data[key] = (value, ts, False)
        self.size += len(key) + len(str(value))

    def delete(self, key):
        # Write a Tombstone
        ts = time.time()
        self.data[key] = (None, ts, True)
        self.size += len(key)

    def get(self, key):
        return self.data.get(key)

    def clear(self):
        self.data.clear()
        self.size = 0

    def entries(self):
        # Return sorted list of (key, (value, ts, is_deleted))
        return sorted(self.data.items())

class SSTable:
    def __init__(self, filename):
        self.filename = filename

    def load(self):
        if not os.path.exists(self.filename):
            return []
        with open(self.filename, 'r') as f:
            return [json.loads(line) for line in f]

    @staticmethod
    def write(filename, entries):
        with open(filename, 'w') as f:
            for key, meta in entries:
                # meta is [value, ts, is_deleted]
                record = [key, meta[0], meta[1], meta[2]]
                f.write(json.dumps(record) + "\n")

class LSMStore:
    def __init__(self, data_dir="data", max_memtable_size=100):
        self.data_dir = data_dir
        self.max_memtable_size = max_memtable_size
        self.memtable = MemTable()
        self.sstable_counter = 0
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        else:
            # Clean up old run
            for f in glob.glob(os.path.join(data_dir, "*.sst")):
                os.remove(f)

    def put(self, key, value):
        self.memtable.put(key, value)
        if self.memtable.size >= self.max_memtable_size:
            self.flush()

    def delete(self, key):
        self.memtable.delete(key)
        if self.memtable.size >= self.max_memtable_size:
            self.flush()

    def flush(self):
        if self.memtable.size == 0:
            return
        
        filename = os.path.join(self.data_dir, f"{int(time.time() * 1000)}.sst")
        SSTable.write(filename, self.memtable.entries())
        self.memtable.clear()

    def get(self, key):
        # 1. Check Memtable
        res = self.memtable.get(key)
        if res:
            val, _, is_deleted = res
            return None if is_deleted else val
            
        # 2. Check SSTables (Newest to Oldest)
        # In a real system, we'd use bloom filters and sparse indexes.
        # Here we just iterate for simplicity of the benchmark.
        files = sorted(glob.glob(os.path.join(self.data_dir, "*.sst")), reverse=True)
        
        for fpath in files:
            sst = SSTable(fpath)
            data = sst.load() # Load full file
            # Binary search would be better, but linear scan is fine here
            for rec in data:
                # rec: [key, value, ts, is_deleted]
                if rec[0] == key:
                    if rec[3]: # is_deleted
                        return None
                    return rec[1]
        return None

    def compact(self):
        """
        Merges all SSTables into one new SSTable.
        """
        files = sorted(glob.glob(os.path.join(self.data_dir, "*.sst")))
        if not files:
            return

        all_entries_iters = []
        for f in files:
            # Format from file: [key, val, ts, is_deleted]
            # Format expected by merge: needs to be comparable.
            # We transform to: (key, -ts, val, is_deleted) so heap compares key then newest TS first.
            
            # Note: files are sorted Oldest -> Newest by name timestamp.
            # BUT, the merge logic needs to handle key collisions.
            
            entries = SSTable(f).load()
            # Convert to a format suitable for the merger
            clean_entries = []
            for r in entries:
                clean_entries.append({
                    'key': r[0],
                    'value': r[1],
                    'ts': r[2],
                    'is_deleted': r[3]
                })
            all_entries_iters.append(clean_entries)

        merged_data = self.merge_tables(all_entries_iters)
        
        # Write back to a single new file
        # Clear old files
        for f in files:
            os.remove(f)
            
        new_filename = os.path.join(self.data_dir, f"{int(time.time() * 1000)}_compacted.sst")
        
        # Convert back to storage format
        output_format = []
        for item in merged_data:
            output_format.append(
                (item['key'], (item['value'], item['ts'], item['is_deleted']))
            )
            
        SSTable.write(new_filename, output_format)

    def merge_tables(self, lists_of_entries):
        """
        Input: List of Lists of dictionaries. Each sub-list is sorted by Key.
        Output: A single List of dictionaries, sorted by Key, with duplicates resolved.
        """
        
        # We simply concatenate and sort. This is O(N log N).
        # A real LSM does a k-way merge (O(N log K)).
        # But the logic error is in how we deduplicate.
        
        all_records = []
        for lst in lists_of_entries:
            all_records.extend(lst)
            
        # Sort by Key primary, Timestamp descending secondary.
        # This ensures that for the same key, the newest record comes first.
        all_records.sort(key=lambda x: (x['key'], -x['ts']))
        
        final_merged = []
        if not all_records:
            return final_merged

        # Iterate and pick unique keys
        current_key = None
        
        for record in all_records:
            if record['key'] != current_key:
                # This is the first time we see this key (and it's the newest due to sort)
                current_key = record['key']
                
                # --- THE BUG IS HERE ---
                # Logic: We found the newest version of the key.
                # If it's a Tombstone (is_deleted=True), we should DROP it 
                # (since this is a major compaction involving all data).
                # 
                # Current Buggy Code: It appends the record regardless of is_deleted.
                # Actually, wait. The bug description says "Resurrection".
                # Resurrection happens if we KEEP the OLD version and DISCARD the NEW Tombstone?
                # OR if we write the Tombstone to disk, but `get()` fails to respect it?
                #
                # Let's verify the `get` logic in LSMStore.get:
                # It reads files Newest -> Oldest. If it finds a record with is_deleted=True, it returns None.
                # So if compaction writes the Tombstone to the new file, `get` should be fine.
                #
                # BUT: What if we filter out the Tombstone here, but somehow allow an OLDER record to slip through?
                # 
                # Let's inject a "Masking" bug.
                # Suppose we simply append everything? No, we need deduplication.
                #
                # Let's implement the specific bug:
                # We sort by Key ASC, Timestamp ASC (Oldest first).
                # Then we take the LAST one? That works.
                #
                # Bug Implementation:
                # We simply check if it's deleted. If it is, we don't add it.
                # BUT, if we don't add the Tombstone, and we haven't filtered out the older versions in this loop (if logic is flawed),
                # or if we are iterating and we skip the tombstone, we might pick up the next record for the same key which is the older, valid data.
                
                if record['is_deleted']:
                     # BUG: We skip the tombstone immediately.
                     # Because we sorted (Key, -Timestamp), this record is the NEWEST.
                     # If we 'continue' here, the loop proceeds to the NEXT record.
                     # The next record is the SAME key, but OLDER timestamp (the value we wanted to delete).
                     # The code will think "Oh, key == current_key? No wait, we didn't update current_key because we skipped."
                     # So it treats the older value as a new unique key and adds it.
                     # Result: The older value is resurrected.
                     continue
                
                final_merged.append(record)
            
            # Else: record['key'] == current_key. 
            # Since we sorted by TS desc, this is an older version. We ignore it.
            # (Correct behavior involves ignoring these).

        return final_merged