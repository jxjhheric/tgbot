package main

import "testing"

func TestMukakuResourceLabel(t *testing.T) {
	tests := []struct {
		name     string
		resource MukakuResource
		want     string
	}{
		{
			name: "simple subtitles",
			resource: MukakuResource{
				Name: "阿凡达：火与烬[简繁英字幕].Avatar.Fire.and.Ash.2025.2160p.WEB-DL.H.265.HDR",
				Size: "34.55 GB",
			},
			want: "2160P · H.265 · 简繁英字幕 · 34.55 GB",
		},
		{
			name: "mixed audio and subtitles",
			resource: MukakuResource{
				Name: "阿凡达：火与烬[国英多音轨+特效中文字幕].2025.1080p.BluRay.x265.10bit",
				Size: "21.73 GB",
			},
			want: "1080P · X265 · 特效中文字幕 · 21.73 GB",
		},
		{
			name: "quality fallback",
			resource: MukakuResource{
				Name:    "Example.Movie.WEB-DL.AV1",
				Quality: "WEB-4K",
				Size:    "8 GB",
			},
			want: "4K · AV1 · 8 GB",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := mukakuResourceLabel(test.resource); got != test.want {
				t.Fatalf("mukakuResourceLabel() = %q, want %q", got, test.want)
			}
		})
	}
}
