def _decode_full_report(self, headers, contact_data) -> list[dict[str, object]]:
    decoded: list[dict[str, object]] = []
    contact_data_count = len(contact_data)
    for header in headers:
        offset = int(header.contact_data_offset)
        count = int(header.num_contact_data)
        if offset < 0 or offset + count > contact_data_count:
            raise RuntimeError("contact header data range is invalid")
        paths = tuple(
            self._decode_path(value)
            for value in (
                header.actor0,
                header.actor1,
                header.collider0,
                header.collider1,
            )
        )
        contacts = []
        for index in range(offset, offset + count):
            record = contact_data[index]
            position,normal,impulse=record.position,record.normal,record.impulse
            contacts.append({
                # Native Float3 slices own their floats; tuples retain immutable vector copies.
                "position_m": tuple(position[:3]),
                "normal": tuple(normal[:3]),
                "impulse_n_s": tuple(impulse[:3]),
                "separation_m": float(record.separation),
            })
        decoded.append({
            "paths": paths,
            "records": count,
            "contact_data_offset": offset,
            "contacts": contacts,
        })
    return decoded
